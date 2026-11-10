/*
 * Independent certificate checker for eight alternatives and three voters.
 *
 * This program checks supplied certificates. It does not search for rankings
 * and does not solve a linear program. All mathematical checks use integers.
 *
 * Compile:
 *   g++ -std=c++11 -O3 -Wall -Wextra -pedantic check_n8_k3.cpp -o checker
 *
 * Run with: ./checker tourn8.txt inequalities.txt certificates.txt
 *
 * The code uses ordinary functions and data records, with no user-defined
 * classes, constructors, member functions, type inference or range-based loops.
 * The C++11 compiler setting is needed for standard long long integer support;
 * the program otherwise uses the older, explicit style of C++.
 *
 * Conventions:
 *   Alternatives are numbered 0,...,7.
 *   Pair coordinates are (0,1), (0,2), ..., (0,7), (1,2), ..., (6,7).
 *   Coordinate (a,b), a<b, counts voters who place a BEFORE b.
 *   Ranking IDs are 0,...,40319, in lexicographic permutation order.
 *   Inequality IDs are zero-based row numbers in the inequality file.
 *
 * Certificate-file modes:
 *   "sample N": check N specified candidates. This is only a sample check.
 *   "full": enumerate every candidate and require its certificate in order.
 *
 * The supplied tournament catalogue must contain a representative of every
 * isomorphism class. This program checks its format and duplicate encodings;
 * it does NOT prove the completeness of that catalogue.
 */

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

const int NUMBER_OF_ALTERNATIVES = 8;
const int NUMBER_OF_VOTERS = 3;
const int NUMBER_OF_PAIRS = 28;       // 8 * 7 / 2.
const int NUMBER_OF_TRIPLES = 56;     // 8 * 7 * 6 / (3 * 2 * 1).
const int NUMBER_OF_RANKINGS = 40320; // 8 factorial.
const int NUMBER_OF_TOURNAMENTS = 6880;
const unsigned long EXPECTED_SURVIVOR_COUNT = 25361255UL;

typedef long long ExactInteger;
const ExactInteger MAXIMUM_COEFFICIENT = 1000000000LL;
const ExactInteger MAXIMUM_BOUND = 28000000000LL;

// A PairSet represents a subset of the 28 pairs. Its implementation is private
// to the small set functions below. Elsewhere, use those functions by name.
typedef unsigned long PairSet;

// These structs contain data only: they have no constructors or methods.
struct AlternativePair {
    int firstAlternative;
    int secondAlternative;
};

struct AlternativeTriple {
    // For alternatives a<b<c, these are the indices of (a,b), (b,c), (a,c).
    int firstSecondPair;
    int secondThirdPair;
    int firstThirdPair;
};

struct ComparisonVector {
    int counts[NUMBER_OF_PAIRS];
};

struct SeparatingInequality {
    ExactInteger coefficients[NUMBER_OF_PAIRS];
    ExactInteger bound;
};

struct Tournament {
    std::string encoding;
    // Starting comparison vector: every majority is provisionally 2–1.
    // For each stored pair a<b, the entry counts voters placing a before b:
    // 2 if a beats b in the tournament, and 1 if b beats a.
    // Selecting a pair for unanimity later changes 2 to 3, or 1 to 0.
    ComparisonVector majorityCounts; // Counts 1 or 2, before choosing unanimity.
    PairSet eligibleUnanimousPairs;
};

struct VerificationTotals {
    unsigned long positiveCertificates;
    unsigned long negativeCertificates;
};

// The four tables below are initialized once and then only read.
// Functions that use a table name that dependency in their comments.
AlternativePair alternativePairs[NUMBER_OF_PAIRS];
AlternativeTriple alternativeTriples[NUMBER_OF_TRIPLES];
std::vector<ComparisonVector> rankingVectors;
std::vector<SeparatingInequality> verifiedInequalities;


/* Purpose: stop immediately if a required condition is false.
 * Receives: a condition and an explanation to print if it is false.
 * Returns: nothing. If the condition is false, prints an error and exits
 *          with failure status; execution cannot continue past that error.
 */
void requireCondition(bool condition, const char *explanation)
{
    if (!condition) {
        std::cerr << "CHECK FAILED: " << explanation << '\n';
        std::exit(EXIT_FAILURE);
    }
}


/* Purpose: construct the empty set of pairs.
 * Receives: nothing.
 * Returns: a PairSet containing no pair.
 */
PairSet emptyPairSet()
{
    return 0UL;
}


/* Purpose: decide whether a particular pair belongs to a set.
 * Receives: a set and a pair index between 0 and 27.
 * Returns: true exactly when that pair is present. Changes nothing.
 */
bool pairIsPresent(PairSet pairSet, int pairIndex)
{
    // Bit pairIndex records membership of that pair. All other bits are ignored.
    return (pairSet & (1UL << pairIndex)) != 0UL;
}


/* Purpose: add one pair to a set.
 * Receives: a set and a pair index between 0 and 27.
 * Returns: the set with that pair included. The argument itself is unchanged.
 */
PairSet includePair(PairSet pairSet, int pairIndex)
{
    return pairSet | (1UL << pairIndex);
}


/* Purpose: remove one pair from a set.
 * Receives: a set and a pair index between 0 and 27.
 * Returns: the set with that pair absent. Removing an absent pair is harmless.
 */
PairSet excludePair(PairSet pairSet, int pairIndex)
{
    return pairSet & ~(1UL << pairIndex);
}


/* Purpose: test containment of sets of pairs.
 * Receives: a proposed subset and a proposed containing set.
 * Returns: true exactly when every pair in the first belongs to the second.
 */
bool pairSetIsSubset(PairSet proposedSubset, PairSet containingSet)
{
    // Look for a bit present in the proposed subset but absent from the other set.
    return (proposedSubset & ~containingSet) == 0UL;
}


/* Purpose: interpret a subset number as choices from an ordered list.
 * Receives: a subset number and a list position between 0 and 27.
 * Returns: true when that position is selected in the subset.
 * Here bit positions refer to list positions, not directly to pair indices.
 */
bool subsetSelectsPosition(unsigned long subsetNumber, int positionInList)
{
    return (subsetNumber & (1UL << positionInList)) != 0UL;
}


/* Purpose: count the subsets of a set with a given number of elements.
 * Receives: an element count between 0 and 28.
 * Returns: 2 raised to that count, including the empty subset.
 */
unsigned long numberOfSubsets(int elementCount)
{
    return 1UL << elementCount;
}


/* Purpose: construct the set containing all 28 pairs.
 * Receives: nothing.
 * Returns: a PairSet with every pair present.
 */
PairSet allPairsSet()
{
    PairSet result = emptyPairSet();
    for (int pairIndex = 0; pairIndex < NUMBER_OF_PAIRS; ++pairIndex) {
        result = includePair(result, pairIndex);
    }
    return result;
}


/* Purpose: give each pair and each triple a fixed index.
 * Receives: nothing.
 * Returns: nothing. Fills alternativePairs and alternativeTriples.
 * Must run once, before any function that reads those tables.
 */
void initializePairsAndTriples()
{
    int pairIndexOfAlternatives[NUMBER_OF_ALTERNATIVES][NUMBER_OF_ALTERNATIVES];
    int nextPairIndex = 0;

    for (int first = 0; first < NUMBER_OF_ALTERNATIVES; ++first) {
        for (int second = first + 1; second < NUMBER_OF_ALTERNATIVES; ++second) {
            alternativePairs[nextPairIndex].firstAlternative = first;
            alternativePairs[nextPairIndex].secondAlternative = second;
            pairIndexOfAlternatives[first][second] = nextPairIndex;
            ++nextPairIndex;
        }
    }
    requireCondition(nextPairIndex == NUMBER_OF_PAIRS, "incorrect number of pairs");

    int nextTripleIndex = 0;
    for (int first = 0; first < NUMBER_OF_ALTERNATIVES; ++first) {
        for (int second = first + 1; second < NUMBER_OF_ALTERNATIVES; ++second) {
            for (int third = second + 1; third < NUMBER_OF_ALTERNATIVES; ++third) {
                AlternativeTriple triple;
                triple.firstSecondPair = pairIndexOfAlternatives[first][second];
                triple.secondThirdPair = pairIndexOfAlternatives[second][third];
                triple.firstThirdPair = pairIndexOfAlternatives[first][third];
                alternativeTriples[nextTripleIndex] = triple;
                ++nextTripleIndex;
            }
        }
    }
    requireCondition(nextTripleIndex == NUMBER_OF_TRIPLES, "incorrect number of triples");
}


/* Purpose: construct the comparison vector of every possible ranking.
 * Receives: nothing; reads the initialized alternativePairs table.
 * Returns: nothing. Fills rankingVectors in lexicographic permutation order.
 * Thus the vector at index i has ranking ID i in all certificate files.
 */
void enumerateAllRankings()
{
    int permutation[NUMBER_OF_ALTERNATIVES];
    for (int position = 0; position < NUMBER_OF_ALTERNATIVES; ++position) {
        permutation[position] = position;
    }

    // The initial permutation is 0,1,...,7. next_permutation visits every
    // permutation once, in lexicographic order, and returns false at the end.
    do {
        int positionOfAlternative[NUMBER_OF_ALTERNATIVES];
        for (int position = 0; position < NUMBER_OF_ALTERNATIVES; ++position) {
            positionOfAlternative[permutation[position]] = position;
        }

        ComparisonVector ranking;
        for (int pairIndex = 0; pairIndex < NUMBER_OF_PAIRS; ++pairIndex) {
            int first = alternativePairs[pairIndex].firstAlternative;
            int second = alternativePairs[pairIndex].secondAlternative;
            if (positionOfAlternative[first] < positionOfAlternative[second]) {
                ranking.counts[pairIndex] = 1;
            } else {
                ranking.counts[pairIndex] = 0;
            }
        }
        rankingVectors.push_back(ranking);
    } while (std::next_permutation(permutation, permutation + NUMBER_OF_ALTERNATIVES));

    requireCondition(rankingVectors.size() == NUMBER_OF_RANKINGS,
                     "incorrect number of rankings");
}


/* Purpose: compute the left-hand side of a supplied inequality.
 * Receives: a comparison vector with coordinates 0,...,3, and an inequality
 *           with range-checked integers.
 * Returns: the exact integer sum of coefficient[p] * counts[p] over all pairs.
 * Changes nothing. The bound is not used in this calculation.
 */
ExactInteger inequalityLeftHandSide(const ComparisonVector &comparisons,
                                   const SeparatingInequality &inequality)
{
    ExactInteger result = 0;
    for (int pairIndex = 0; pairIndex < NUMBER_OF_PAIRS; ++pairIndex) {
        result += inequality.coefficients[pairIndex] * comparisons.counts[pairIndex];
    }
    return result;
}


/* Purpose: ensure an input contains no unexpected trailing data.
 * Receives: an input stream and the explanation to use if this check fails.
 * Returns: nothing. Skips trailing whitespace, then requires ordinary EOF.
 * A read error or any remaining non-whitespace token causes failure.
 */
void requireEndOfInput(std::istream &input, const char *explanation)
{
    std::string unexpectedToken;
    bool foundAnotherToken = bool(input >> unexpectedToken);
    requireCondition(!foundAnotherToken && input.eof() && !input.bad(), explanation);
}


/* Purpose: read the inequality table and prove every row valid on all rankings.
 * Receives: the inequality input stream; reads the initialized rankingVectors.
 * Returns: nothing. Fills verifiedInequalities, preserving the file's row order.
 * Stops on malformed input, unsupported integer sizes, or an invalid inequality.
 */
void readAndVerifyAllInequalities(std::istream &input)
{
    int inequalityCount;
    requireCondition(bool(input >> inequalityCount) && inequalityCount >= 0,
                     "missing or invalid inequality count");

    for (int inequalityIndex = 0; inequalityIndex < inequalityCount; ++inequalityIndex) {
        SeparatingInequality inequality;
        requireCondition(bool(input >> inequality.bound), "missing inequality bound");
        requireCondition(inequality.bound >= -MAXIMUM_BOUND &&
                         inequality.bound <= MAXIMUM_BOUND,
                         "inequality bound exceeds the supported integer range");

        for (int pairIndex = 0; pairIndex < NUMBER_OF_PAIRS; ++pairIndex) {
            ExactInteger coefficient;
            requireCondition(bool(input >> coefficient), "missing inequality coefficient");
            requireCondition(coefficient >= -MAXIMUM_COEFFICIENT &&
                             coefficient <= MAXIMUM_COEFFICIENT,
                             "inequality coefficient exceeds the supported integer range");
            inequality.coefficients[pairIndex] = coefficient;
        }

        // Ranking coordinates are 0 or 1; candidate coordinates are 0,...,3.
        // Therefore every left-hand side has absolute value at most
        // 28 * 10^9 * 3 = 84 billion. Three times the permitted bound has
        // that same maximum magnitude. These values fit in ExactInteger.
        for (int rankingIndex = 0; rankingIndex < NUMBER_OF_RANKINGS; ++rankingIndex) {
            ExactInteger value = inequalityLeftHandSide(rankingVectors[rankingIndex],
                                                       inequality);
            requireCondition(value <= inequality.bound,
                             "supplied inequality fails on a ranking");
        }
        verifiedInequalities.push_back(inequality);
    }
    requireEndOfInput(input, "trailing data or read error in inequality file");
}


/* Purpose: read the majority directions from a tournament encoding and
 *          prepare the data used to construct candidate comparison vectors.
 *
 * Receives: a 28-character binary encoding, one character per pair (a,b),
 *           with a<b. Reads the initialized alternativeTriples table.
 *
 * Returns: a Tournament containing the encoding, starting comparison counts,
 *          and the set of pairs not ruled out for unanimity by majority cycles.
 *
 * Meaning of the encoding and starting counts:
 *   The encoding specifies directions, not voter counts.
 *   For each pair (a,b), a<b, the stored count always counts voters placing
 *   a before b. We initially assign a non-unanimous, 2-to-1 majority:
 *
 *     encoding '1': majority direction a->b; store count 2.
 *     encoding '0': majority direction b->a; store count 1.
 *
 *   In the second case, storing 1 for a before b means that the opposite
 *   count, for b before a, is 3 - 1 = 2.
 *
 *   These starting counts are not claimed to be realizable by three rankings.
 *   Later, buildCandidate chooses some pairs to be unanimous, changing their
 *   stored counts from 2 to 3 or from 1 to 0. Both changes preserve the
 *   majority direction.
 *
 * Determining which pairs remain eligible for unanimity:
 *   Initially, include all 28 pairs. Then examine every triple.
 *   If its majority directions form a directed cycle, remove its three pairs
 *   from the eligible set. Along the cycle's arrows, each count is at least 2;
 *   the triangle inequality limits their sum to 6, so each must be exactly 2.
 *   Thus none of these comparisons can be unanimous.
 *
 *   "Eligible" means only that this cycle test has not ruled the pair out.
 *   Later checks may still reject candidates choosing that pair as unanimous.
 */
Tournament decodeTournament(const std::string &encoding)
{
    requireCondition(encoding.size() == NUMBER_OF_PAIRS,
                     "tournament encoding must contain 28 bits");
    Tournament tournament;
    tournament.encoding = encoding;
    tournament.eligibleUnanimousPairs = allPairsSet();

    for (int pairIndex = 0; pairIndex < NUMBER_OF_PAIRS; ++pairIndex) {
        requireCondition(encoding[pairIndex] == '0' || encoding[pairIndex] == '1',
                         "tournament encoding contains a character other than 0 or 1");
        if (encoding[pairIndex] == '1') {
            tournament.majorityCounts.counts[pairIndex] = 2;
        } else {
            tournament.majorityCounts.counts[pairIndex] = 1;
        }
    }

    for (int tripleIndex = 0; tripleIndex < NUMBER_OF_TRIPLES; ++tripleIndex) {
        AlternativeTriple triple = alternativeTriples[tripleIndex];
        int firstSecondCount = tournament.majorityCounts.counts[triple.firstSecondPair];
        int secondThirdCount = tournament.majorityCounts.counts[triple.secondThirdPair];
        int firstThirdCount = tournament.majorityCounts.counts[triple.firstThirdPair];

        // For a<b<c there are precisely two directed majority cycles:
        // a->b->c->a gives these three stored counts as (2,2,1);
        // a<-b<-c<-a gives them as (1,1,2).
        bool isMajorityCycle = firstSecondCount == secondThirdCount &&
                               secondThirdCount != firstThirdCount;
        if (isMajorityCycle) {
            // Along a directed majority cycle, each cyclic count is at least 2.
            // Their sum is at most 6 by a triangle inequality, so all are 2.
            // Raising any one to the unanimous count 3 is impossible.
            tournament.eligibleUnanimousPairs = excludePair(
                tournament.eligibleUnanimousPairs, triple.firstSecondPair);
            tournament.eligibleUnanimousPairs = excludePair(
                tournament.eligibleUnanimousPairs, triple.secondThirdPair);
            tournament.eligibleUnanimousPairs = excludePair(
                tournament.eligibleUnanimousPairs, triple.firstThirdPair);
        }
    }
    return tournament;
}


/* Purpose: read and check the format of the tournament catalogue.
 * Receives: the catalogue stream; reads alternativeTriples via decodeTournament.
 * Returns: the tournaments in file order. Also fills the supplied encoding set
 *          for later sample-membership checks. Stops on duplicate encodings,
 *          malformed data, read errors or an incorrect number of entries.
 * This does not check isomorphism or prove catalogue completeness.
 */
std::vector<Tournament> readTournamentCatalogue(std::istream &input,
                                              std::set<std::string> &catalogueEncodings)
{
    std::vector<Tournament> tournaments;
    std::string encoding;
    while (input >> encoding) {
        requireCondition(catalogueEncodings.count(encoding) == 0,
                         "duplicate tournament encoding in catalogue");
        catalogueEncodings.insert(encoding);
        tournaments.push_back(decodeTournament(encoding));
    }
    requireCondition(input.eof() && !input.bad(), "read error in tournament catalogue");
    requireCondition(tournaments.size() == NUMBER_OF_TOURNAMENTS,
                     "catalogue must contain 6880 tournaments");
    return tournaments;
}


/* Purpose: form the target comparison vector for a tournament and unanimity set.
 * Receives: a decoded tournament and the set of pairs chosen to be unanimous.
 * Returns: counts 1 or 2 on other pairs, changed to 0 or 3 on unanimous pairs.
 * Stops if any selected pair is ineligible. This function does not yet check
 * the triangle inequalities or any certificate.
 */
ComparisonVector buildCandidate(const Tournament &tournament, PairSet unanimousPairs)
{
    requireCondition(pairSetIsSubset(unanimousPairs, tournament.eligibleUnanimousPairs),
                     "candidate selects an ineligible unanimous pair");
    ComparisonVector candidate = tournament.majorityCounts;
    for (int pairIndex = 0; pairIndex < NUMBER_OF_PAIRS; ++pairIndex) {
        if (pairIsPresent(unanimousPairs, pairIndex)) {
            if (candidate.counts[pairIndex] == 2) {
                candidate.counts[pairIndex] = NUMBER_OF_VOTERS;
            } else {
                candidate.counts[pairIndex] = 0;
            }
        }
    }
    return candidate;
}


/* Purpose: test all triangle inequalities on aggregate pairwise counts.
 * Receives: a candidate vector; reads the initialized alternativeTriples.
 * Returns: true exactly when 0 <= c(a,b)+c(b,c)-c(a,c) <= 3 for all a<b<c.
 * Returns false immediately on the first failure. Changes nothing.
 */
bool passesAllTriangleInequalities(const ComparisonVector &candidate)
{
    for (int tripleIndex = 0; tripleIndex < NUMBER_OF_TRIPLES; ++tripleIndex) {
        AlternativeTriple triple = alternativeTriples[tripleIndex];
        int triangleValue = candidate.counts[triple.firstSecondPair]
                          + candidate.counts[triple.secondThirdPair]
                          - candidate.counts[triple.firstThirdPair];
        if (triangleValue < 0 || triangleValue > NUMBER_OF_VOTERS) {
            return false;
        }
    }
    return true;
}


/* Purpose: verify the three rankings in a positive certificate.
 * Receives: the target candidate and a stream positioned just after its 'P'.
 * Reads: the initialized rankingVectors table.
 * Returns: nothing; stops unless the next three valid ranking IDs have vectors
 *          whose coordinatewise sum equals the target exactly.
 * Repeated ranking IDs are allowed: different voters may have the same ranking.
 */
void verifyPositiveCertificate(const ComparisonVector &candidate, std::istream &input)
{
    ComparisonVector sumOfRankings;
    for (int pairIndex = 0; pairIndex < NUMBER_OF_PAIRS; ++pairIndex) {
        sumOfRankings.counts[pairIndex] = 0;
    }

    for (int voterIndex = 0; voterIndex < NUMBER_OF_VOTERS; ++voterIndex) {
        int rankingId;
        requireCondition(bool(input >> rankingId), "missing ranking ID");
        requireCondition(rankingId >= 0 && rankingId < NUMBER_OF_RANKINGS,
                         "ranking ID is outside 0,...,40319");
        for (int pairIndex = 0; pairIndex < NUMBER_OF_PAIRS; ++pairIndex) {
            sumOfRankings.counts[pairIndex] += rankingVectors[rankingId].counts[pairIndex];
        }
    }

    for (int pairIndex = 0; pairIndex < NUMBER_OF_PAIRS; ++pairIndex) {
        requireCondition(sumOfRankings.counts[pairIndex] == candidate.counts[pairIndex],
                         "three supplied rankings do not sum to the candidate");
    }
}


/* Purpose: verify a negative certificate that excludes fractional realizability.
 * Receives: the target candidate and a stream positioned just after its 'N'.
 * Reads: verifiedInequalities, already checked on every ranking.
 * Returns: nothing; stops unless the next valid inequality ID has alpha.c > 3*beta.
 * This contradicts the bound for any nonnegative weighted sum of rankings with
 * total weight three, so it excludes fractional as well as integer realization.
 */
void verifyNegativeCertificate(const ComparisonVector &candidate, std::istream &input)
{
    int inequalityId;
    requireCondition(bool(input >> inequalityId), "missing inequality ID");
    requireCondition(inequalityId >= 0 && inequalityId < (int)verifiedInequalities.size(),
                     "inequality ID is outside the supplied table");
    const SeparatingInequality &inequality = verifiedInequalities[inequalityId];
    ExactInteger candidateValue = inequalityLeftHandSide(candidate, inequality);
    ExactInteger permittedBound = NUMBER_OF_VOTERS * inequality.bound;
    requireCondition(candidateValue > permittedBound,
                     "inequality does not exclude this candidate");
}


/* Purpose: read and verify one positive or negative certificate.
 * Receives: the target, the certificate stream and running certificate totals.
 * Returns: nothing. Consumes one certificate, invokes the appropriate checker
 *          above, and increases its total only after successful verification.
 * A record is either 'P rankingId rankingId rankingId' or 'N inequalityId'.
 */
void verifyOneCertificate(const ComparisonVector &candidate, std::istream &input,
                          VerificationTotals &totals)
{
    char certificateKind;
    requireCondition(bool(input >> certificateKind), "missing certificate");
    if (certificateKind == 'P') {
        verifyPositiveCertificate(candidate, input);
        ++totals.positiveCertificates;
    } else if (certificateKind == 'N') {
        verifyNegativeCertificate(candidate, input);
        ++totals.negativeCertificates;
    } else {
        requireCondition(false, "certificate must begin with P or N");
    }
}


/* Purpose: list a tournament's eligible unanimous pairs in increasing index order.
 * Receives: a decoded tournament.
 * Returns: the list of pair indices. Changes nothing.
 * The list order determines how subset numbers are interpreted below.
 */
std::vector<int> listEligiblePairs(const Tournament &tournament)
{
    std::vector<int> eligiblePairIndices;
    for (int pairIndex = 0; pairIndex < NUMBER_OF_PAIRS; ++pairIndex) {
        if (pairIsPresent(tournament.eligibleUnanimousPairs, pairIndex)) {
            eligiblePairIndices.push_back(pairIndex);
        }
    }
    return eligiblePairIndices;
}


/* Purpose: translate a subset number into the actual selected pair indices.
 * Receives: the eligible-pair list and a subset number from 0 to 2^listSize-1.
 * Returns: the set of unanimous pairs selected by that subset number.
 * Example: for eligible list [2,9,17], subset number 5 selects positions 0 and 2,
 * so the returned pair set contains pairs 2 and 17. The list itself is unchanged.
 */
PairSet unanimousPairsForSubset(const std::vector<int> &eligiblePairIndices,
                               unsigned long subsetNumber)
{
    PairSet unanimousPairs = emptyPairSet();
    for (int positionInList = 0; positionInList < (int)eligiblePairIndices.size();
         ++positionInList) {
        if (subsetSelectsPosition(subsetNumber, positionInList)) {
            int selectedPairIndex = eligiblePairIndices[positionInList];
            unanimousPairs = includePair(unanimousPairs, selectedPairIndex);
        }
    }
    return unanimousPairs;
}

/* Purpose: enumerate all candidates and verify a certificate for every survivor.
 * Receives: the catalogue in file order, a stream just after having read 'full', and totals.
 * Returns: nothing. Updates totals and stops on any missing or invalid certificate.
 * Reads the initialized triple, ranking and inequality tables via functions above.
 * Coverage is up to relabelling, conditional on catalogue completeness.
 *
 * Why these are all the candidates: with three voters, each pair has a unique
 * majority direction. The count in that direction is either 2 or 3. Fixing the
 * tournament therefore leaves just the choice of which pairs have count 3.
 * Pairs on majority cycles were excluded from that choice in decodeTournament;
 * those choices would already violate a triangle inequality. We enumerate every
 * remaining choice and test its triangle inequalities explicitly.
 */
void verifyEveryCandidate(const std::vector<Tournament> &tournaments,
                          std::istream &input, VerificationTotals &totals)
{
    for (int tournamentIndex = 0; tournamentIndex < (int)tournaments.size();
         ++tournamentIndex) {
        const Tournament &tournament = tournaments[tournamentIndex];
        std::vector<int> eligiblePairIndices = listEligiblePairs(tournament);
        unsigned long subsetCount = numberOfSubsets((int)eligiblePairIndices.size());

        // Every subset is visited once, including the empty subset. The candidate
        // is generated here; a certificate cannot substitute a different target.
        for (unsigned long subsetNumber = 0; subsetNumber < subsetCount; ++subsetNumber) {
            PairSet unanimousPairs = unanimousPairsForSubset(eligiblePairIndices, subsetNumber);
            ComparisonVector candidate = buildCandidate(tournament, unanimousPairs);
            if (passesAllTriangleInequalities(candidate)) {
                verifyOneCertificate(candidate, input, totals);
            }
            // A triangle failure already excludes membership in 3 LO_8 and
            // therefore requires no certificate. Enumeration still continues.
        }
    }

    // This is a diagnostic census check, not a substitute for enumeration.
    unsigned long checkedCandidates = totals.positiveCertificates + totals.negativeCertificates;
    requireCondition(checkedCandidates == EXPECTED_SURVIVOR_COUNT,
                     "exhaustive survivor count differs from 25361255");
}


/* Purpose: initialize the tables, verify the supplied files and report success.
 * Receives: three command-line paths: tournament catalogue, inequalities,
 *           certificates. See the compile/run instructions at the top.
 * Returns: EXIT_SUCCESS after every required check passes. Any failure prints
 *          an explanation and terminates with EXIT_FAILURE through requireCondition.
 * CPU time is reported for this process; it is not a wall-clock measurement.
 */
int main(int argumentCount, char *arguments[])
{
    requireCondition(argumentCount == 4,
                     "usage: checker catalogue inequalities certificates");
    requireCondition(LLONG_MAX >= 84000000000LL,
                     "this platform cannot represent the required exact integers");
    requireCondition(sizeof(PairSet) * CHAR_BIT > NUMBER_OF_PAIRS,
                     "this platform cannot represent sets of 28 pairs");
    std::clock_t startingCpuTime = std::clock();

    initializePairsAndTriples();
    enumerateAllRankings();

    std::ifstream catalogueInput(arguments[1]);
    std::ifstream inequalityInput(arguments[2]);
    std::ifstream certificateInput(arguments[3]);
    requireCondition(catalogueInput.is_open(), "cannot open tournament catalogue");
    requireCondition(inequalityInput.is_open(), "cannot open inequality file");
    requireCondition(certificateInput.is_open(), "cannot open certificate file");

    std::set<std::string> catalogueEncodings;
    std::vector<Tournament> tournaments = readTournamentCatalogue(catalogueInput,
                                                                catalogueEncodings);
    readAndVerifyAllInequalities(inequalityInput);

    VerificationTotals totals;
    totals.positiveCertificates = 0;
    totals.negativeCertificates = 0;

    std::string mode;
    requireCondition(bool(certificateInput >> mode),
                 "missing certificate-file mode");
    requireCondition(mode == "full",
                 "certificate-file mode must be full");

    verifyEveryCandidate(tournaments, certificateInput, totals);
    
    requireEndOfInput(certificateInput, "extra certificates or certificate-file read error");

    std::cout << "CHECK PASSED: mode=" << mode
              << " positive=" << totals.positiveCertificates
              << " negative=" << totals.negativeCertificates
              << " inequalities=" << verifiedInequalities.size() << '\n';

    std::clock_t endingCpuTime = std::clock();
    if (startingCpuTime != (std::clock_t)-1 && endingCpuTime != (std::clock_t)-1) {
        double elapsedCpuSeconds = (double)(endingCpuTime - startingCpuTime) / CLOCKS_PER_SEC;
        std::cout << "CPU seconds: " << elapsedCpuSeconds << '\n';
    }
    return EXIT_SUCCESS;
}
