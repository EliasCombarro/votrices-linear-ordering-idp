#!/usr/bin/env python3
"""Verify the eight-alternative, four-voter counterexample.

MATHEMATICAL OBJECTS
--------------------
We have eight alternatives, labeled 1, 2, ..., 8.  A ranking is a permutation
of these eight labels, listed from first (preferred) to last.

For a ranking pi, its pairwise-comparison vector x^pi has one coordinate for
each pair a<b.  That coordinate is:

    x^pi_ab = 1 if a occurs BEFORE b in pi,
              0 if b occurs BEFORE a in pi.

There are C(8,2) = 28 coordinates, one for each unordered pair.

The matrix H below encodes an INTEGER target vector h using H_ab = h_ab
for a<b and the convention H_ab + H_ba = 4 for a != b.
The claim is that h belongs to 4*LO_8 but cannot be written as the sum
of four ranking vectors.

WHAT THE PROGRAM PROVES
-----------------------
(A) FRACTIONAL REALIZABILITY: The sum of the comparison vectors of eight
    explicitly given rankings is exactly 2*h.  Giving each ranking
    weight 1/2 therefore realizes h with total weight 8*(1/2) = 4.

(B) NO FOUR-VOTER REALIZATION:
    1.  Compute the unanimous comparisons of h.  Every ranking used
        in a four-voter realization must respect them.
    2.  Compute all tight triangle inequalities of h.  Every ranking
        used in a four-voter realization must attain the same bound.
    3.  Enumerate ALL 8! = 40,320 rankings, keeping only those that
        satisfy the restrictions in steps 1 and 2.  Exactly 12 survive.
    4.  Enumerate ALL C(12+4-1,4) = 1,365 multisets of FOUR rankings
        from those 12.  Repeated rankings ARE allowed.  None has sum h.

(C) EXTENSION TO EVERY k >= 5: Check the one extra finite fact used in
    the paper's counting argument: among the 12 eligible rankings, the
    first fractional-certificate ranking Q is the ONLY one that puts
    alternative 3 before 1 before 5.  We also verify H_13=H_15=1.
    These checks, together with the elementary argument explained below,
    extend the counterexample to every degree k >= 4.

WHY THIS IS A COMPLETE PROOF
----------------------------
Any four-voter realization is a multiset of four total rankings.
Every one of those rankings MUST satisfy the unanimous and tight-triangle
conditions; thus all four belong to our explicitly enumerated 12.
Step B4 checks every possible multiset of those 12, and hence excludes
all possible four-voter realizations.  This is a *necessary-condition*
filter followed by a *complete finite search*, not an assumption that
unanimity and triangle inequalities characterize 4*LO_8.

For k >= 5 put h_k = h + (k-4)*x^Q.  The fractional certificate proves
h_k belongs to k*LO_8.  All unanimous and tight triangle conditions
remain valid because Q itself satisfies them.  Every candidate ranking
must thus be among the same 12.  Any eligible ranking other than Q
reverses at least one of 3-before-1 or 1-before-5.  In h_k, exactly
one voter may put 1 before 3 and exactly three may put 5 before 1.
Thus at most four voters differ from Q.  A decomposition of h_k would
have at least k-4 copies of Q, leaving four rankings summing to h,
contrary to part (B).  This is a mathematical argument for every k,
not a claim that infinitely many degrees have been enumerated.

USAGE
-----
    python3 verify_n8_k4.py

For more detail about the intermediate filters:
    python3 verify_n8_k4.py --show-eligible --show-tight

Everything uses ordinary, exact integer arithmetic.  Every essential
check calls require(), which raises an exception on failure.  We do NOT
use the Python 'assert' statement, because 'python3 -O' disables asserts.
"""

import argparse
from itertools import combinations, combinations_with_replacement, permutations
from math import comb, factorial
from time import perf_counter


# ===========================================================================
# 1. INPUT DATA
# ===========================================================================

# We are studying rankings of eight alternatives with four ordinary voters.
NUMBER_OF_ALTERNATIVES = 8
NUMBER_OF_VOTERS = 4

# This is the matrix H displayed in Section 5 of the manuscript.
#
# PAIRWISE_COUNTS[a-1][b-1] = H_ab, the desired number of voters who
# prefer alternative a to alternative b, where a,b are labels in 1..8.
#
# - Python array indices start at 0; mathematical labels start at 1.
# - Every diagonal entry is 0 because we do not compare an alternative
#   against itself.
# - For a != b, H_ab + H_ba must equal NUMBER_OF_VOTERS = 4.
# - The upper-triangular entries, in lexicographic order, form the
#   28-dimensional target vector h.
PAIRWISE_COUNTS = (
    (0, 2, 1, 2, 1, 3, 1, 3),
    (2, 0, 0, 1, 1, 1, 0, 1),
    (3, 4, 0, 2, 1, 2, 1, 3),
    (2, 3, 2, 0, 0, 1, 0, 1),
    (3, 3, 3, 4, 0, 2, 1, 3),
    (1, 3, 2, 3, 2, 0, 0, 1),
    (3, 4, 3, 4, 3, 4, 0, 2),
    (1, 3, 1, 3, 1, 3, 2, 0),
)

# These eight rankings are the EXPLICIT FRACTIONAL CERTIFICATE.
#
# Each row lists all alternatives in order of precedence.  Each has
# weight 1/2.  For example, (3,1,8,...) means 3 precedes 1,
# and 1 precedes 8.  Their eight comparison vectors should sum to
# 2*h, coordinate by coordinate.
FRACTIONAL_CERTIFICATE_RANKINGS = (
    (3, 1, 8, 7, 6, 2, 5, 4),
    (5, 1, 8, 7, 6, 4, 3, 2),
    (5, 3, 8, 7, 4, 2, 1, 6),
    (7, 1, 6, 5, 3, 8, 4, 2),
    (7, 3, 2, 5, 1, 8, 6, 4),
    (7, 5, 4, 3, 1, 8, 6, 2),
    (7, 6, 5, 4, 3, 2, 1, 8),
    (8, 7, 6, 5, 4, 3, 2, 1),
)

# The manuscript also explicitly lists the other four eligible rankings.
# Checking equality with the complete enumerated set catches any mismatch
# between the supplementary certificate and those displayed rankings.
OTHER_ELIGIBLE_RANKINGS = (
    (1, 8, 7, 6, 5, 4, 3, 2),
    (5, 3, 1, 8, 7, 6, 4, 2),
    (7, 5, 3, 1, 8, 6, 4, 2),
    (7, 5, 4, 3, 2, 1, 8, 6),
)

# The first certificate ranking is the Q used in the extension to k>=5.
DISTINGUISHED_RANKING_Q = FRACTIONAL_CERTIFICATE_RANKINGS[0]

# Mathematical alternative labels are 1,2,...,8 (NOT 0,1,...,7).
ALTERNATIVES = tuple(range(1, NUMBER_OF_ALTERNATIVES + 1))

# All 28 unordered pairs (a,b), always with a<b, in lexicographic order.
# This is also the exact coordinate order of our target and ranking vectors.
UNORDERED_PAIRS = tuple(combinations(ALTERNATIVES, 2))

# All 56 unordered triples (a,b,d), always with a<b<d.
# The variable name 'd' is used for the third alternative to avoid
# confusing it with our target vector h.
UNORDERED_TRIPLES = tuple(combinations(ALTERNATIVES, 3))


# ===========================================================================
# 2. HELPER FUNCTIONS
# ===========================================================================

def require(condition, explanation):
    """Stop immediately with a clear error if an essential check fails.

    This is deliberate: unlike 'assert', it continues to work under
    python3 -O.  We never want a broken certificate to appear valid
    merely because checks were disabled.
    """
    if not condition:
        raise RuntimeError("VERIFICATION FAILED: " + explanation)


def matrix_entry(first_alternative, second_alternative):
    """Return H_ab using the mathematical labels a,b in 1..8.

    For example, matrix_entry(2,3) = 0 means that in the target matrix,
    alternative 2 precedes alternative 3 with count zero.
    """
    return PAIRWISE_COUNTS[first_alternative - 1][second_alternative - 1]


def positions_in_ranking(ranking):
    """Return each alternative's POSITION in the ranking.

    If ranking = (8,1,2,...) then result[8]=0, result[1]=1,
    result[2]=2.  The entry at position zero is unused, allowing us
    to index by the mathematical labels 1..8 without subtracting one.

    Smaller position means earlier in the ranking.

    We call this many times during enumeration, so ranking validity
    is checked once separately, in validate_input_data().
    """
    position_by_alternative = [-1] * (NUMBER_OF_ALTERNATIVES + 1)
    for position, alternative in enumerate(ranking):
        position_by_alternative[alternative] = position
    return position_by_alternative


def ranking_comparison(positions, first_alternative, second_alternative):
    """Return a single x^pi_ab (either 0 or 1).

    positions is the output of positions_in_ranking().
    x^pi_ab=1 precisely when a precedes b in the ranking.
    """
    return int(positions[first_alternative] < positions[second_alternative])


def comparison_vector(positions):
    """Return all 28 upper-triangular comparison coordinates of a ranking.

    The result is a tuple of zeros and ones, ordered exactly as
    UNORDERED_PAIRS = ((1,2),(1,3),...,(7,8)).

    To compare a sum of ranking vectors with the target, we only need
    these 28 coordinates; opposite-direction coordinates are redundant.
    """
    return tuple(
        ranking_comparison(positions, a, b)
        for a, b in UNORDERED_PAIRS
    )


def validate_input_data():
    """Validate the entire target matrix and the explicit rankings.

    These checks protect against typos in the manuscript or transcription
    errors in this file before any mathematical conclusion is attempted.
    """
    require(len(PAIRWISE_COUNTS) == NUMBER_OF_ALTERNATIVES,
            "The matrix does not have eight rows")

    for row_number, matrix_row in enumerate(PAIRWISE_COUNTS, start=1):
        require(len(matrix_row) == NUMBER_OF_ALTERNATIVES,
                f"Matrix row {row_number} does not have eight columns")

    for a in ALTERNATIVES:
        require(matrix_entry(a, a) == 0,
                f"Diagonal matrix entry H_{a}{a} is not zero")

    for a, b in UNORDERED_PAIRS:
        forward_count = matrix_entry(a, b)
        reverse_count = matrix_entry(b, a)
        require(isinstance(forward_count, int) and
                isinstance(reverse_count, int),
                f"Non-integer matrix entry for pair ({a},{b})")
        require(0 <= forward_count <= NUMBER_OF_VOTERS and
                0 <= reverse_count <= NUMBER_OF_VOTERS,
                f"Entry outside 0,...,4 for pair ({a},{b})")
        require(forward_count + reverse_count == NUMBER_OF_VOTERS,
                f"H_{a}{b}+H_{b}{a} is not four")

    require(len(UNORDERED_PAIRS) == comb(NUMBER_OF_ALTERNATIVES, 2) == 28,
            "Incorrect 28-coordinate indexing")
    require(len(UNORDERED_TRIPLES) == comb(NUMBER_OF_ALTERNATIVES, 3) == 56,
            "Incorrect 56-triple indexing")

    require(len(FRACTIONAL_CERTIFICATE_RANKINGS) == 8,
            "The fractional certificate must contain eight rankings")
    require(len(OTHER_ELIGIBLE_RANKINGS) == 4,
            "The manuscript must list four additional eligible rankings")

    sorted_labels = list(ALTERNATIVES)
    for certificate_index, ranking in enumerate(
            FRACTIONAL_CERTIFICATE_RANKINGS, start=1):
        require(sorted(ranking) == sorted_labels,
                f"Fractional ranking {certificate_index} is not a permutation")
    for other_index, ranking in enumerate(OTHER_ELIGIBLE_RANKINGS, start=1):
        require(sorted(ranking) == sorted_labels,
                f"Additional ranking {other_index} is not a permutation")
    require(len(set(FRACTIONAL_CERTIFICATE_RANKINGS + OTHER_ELIGIBLE_RANKINGS)) == 12,
            "The twelve displayed eligible rankings are not all distinct")


def target_comparison_vector():
    """Construct h using H_ab for a<b, in the 28-coordinate order."""
    return tuple(matrix_entry(a, b) for a, b in UNORDERED_PAIRS)


# ===========================================================================
# 3. VERIFY FRACTIONAL REALIZABILITY
# ===========================================================================

def verify_fractional_realization(target_vector):
    """Check sum_{eight rankings pi} x^pi = 2*h EXACTLY.

    Every ranking has fractional weight 1/2.  Therefore:

        (1/2) * sum_{eight pi} x^pi = h,
        total fractional weight = 8 * (1/2) = 4.

    We avoid floating point: checking the equivalent integer equality
    sum x^pi = 2*h is exact, with no rounding or solver tolerances.
    """
    number_of_coordinates = len(UNORDERED_PAIRS)  # Should be 28.

    # Sum of the comparison vectors across the EIGHT supplied rankings.
    summed_comparison_counts = [0] * number_of_coordinates

    for ranking in FRACTIONAL_CERTIFICATE_RANKINGS:
        positions = positions_in_ranking(ranking)
        ranking_vector = comparison_vector(positions)

        for coordinate_index, comparison_bit in enumerate(ranking_vector):
            summed_comparison_counts[coordinate_index] += comparison_bit

    # Explicitly compare every one of the 28 coordinates with twice h.
    for coordinate_index, (a, b) in enumerate(UNORDERED_PAIRS):
        actual_sum = summed_comparison_counts[coordinate_index]
        required_sum = 2 * target_vector[coordinate_index]
        require(actual_sum == required_sum,
                f"Fractional realization fails for pair ({a},{b}): "
                f"actual {actual_sum}, required {required_sum}")

    # Eight weights each equal 1/2; multiply by 2 to avoid fractions.
    require(len(FRACTIONAL_CERTIFICATE_RANKINGS) == 2 * NUMBER_OF_VOTERS,
            "The certificate does not have total fractional weight four")

    return number_of_coordinates


# ===========================================================================
# 4. EXTRACT THE NECESSARY RESTRICTIONS ON A HYPOTHETICAL VOTER
# ===========================================================================

def unanimous_comparison_requirements():
    """List comparisons that EVERY voter in a realization must obey.

    If H_ab = 4 for a<b, all four voters must put a before b,
    so each ranking must have x^pi_ab = 1.

    If H_ab = 0, all four voters must put b before a,
    so each ranking must have x^pi_ab = 0.

    Values H_ab=1, 2 or 3 impose NO orientation on an individual voter.
    We store each requirement as (a, b, required_comparison_bit).
    """
    requirements = []
    for a, b in UNORDERED_PAIRS:
        count_ab = matrix_entry(a, b)
        if count_ab == 0:
            requirements.append((a, b, 0))
        elif count_ab == NUMBER_OF_VOTERS:
            requirements.append((a, b, 1))

    return requirements


def tight_triangle_requirements():
    """List all tight triangle constraints on INDIVIDUAL rankings.

    For every triple a<b<d, a total ranking pi satisfies

        x^pi_ab + x^pi_bd - x^pi_ad in {0,1}.

    Why?  The equivalent cyclic expression

        x^pi_ab + x^pi_bd + x^pi_da

    is 1 or 2, and x^pi_da = 1 - x^pi_ad.

    For the target h, define the integer

        target_triangle = H_ab + H_bd - H_ad.

    If h is the sum of FOUR ranking vectors, target_triangle is
    the sum of four separate values, each in {0,1}.  Thus it must
    lie between 0 and 4.  Moreover:

      * If target_triangle = 0, EVERY voter must contribute 0.
      * If target_triangle = 4, EVERY voter must contribute 1.
      * If target_triangle = 1, 2 or 3, this triple alone does not
        force the same contribution from all voters.

    We call a triangle "tight" precisely when target_triangle
    equals 0 or 4.  These are NECESSARY constraints on each voter.

    WORKED EXAMPLE (alternatives 1,2,6):

        H_12=2, H_26=1, H_16=3, so
        H_12 + H_26 - H_16 = 2+1-3 = 0.

    Therefore EVERY hypothetical voter must satisfy

        x^pi_12 + x^pi_26 - x^pi_16 = 0.

    Output entries are (a, b, d, forced_individual_triangle_value).
    """
    requirements = []

    for a, b, d in UNORDERED_TRIPLES:
        target_triangle = (
            matrix_entry(a, b)
            + matrix_entry(b, d)
            - matrix_entry(a, d)
        )

        # We also check that h satisfies all 56 triangle inequalities.
        # If it did not, h could not be fractionally realizable.
        require(0 <= target_triangle <= NUMBER_OF_VOTERS,
                f"Target violates triangle inequality for ({a},{b},{d})")

        if target_triangle == 0:
            requirements.append((a, b, d, 0))
        elif target_triangle == NUMBER_OF_VOTERS:
            requirements.append((a, b, d, 1))

    return requirements


def respects_every_unanimity(positions, requirements):
    """Return True exactly when the ranking respects all unanimities."""
    for a, b, required_bit in requirements:
        if ranking_comparison(positions, a, b) != required_bit:
            return False
    return True


def respects_every_tight_triangle(positions, requirements):
    """Return True exactly when a ranking attains every tight bound.

    For a<b<d we directly calculate x_ab + x_bd - x_ad.
    Each required value is either 0 or 1, as derived above.
    """
    for a, b, d, required_value in requirements:
        actual_value = (
            ranking_comparison(positions, a, b)
            + ranking_comparison(positions, b, d)
            - ranking_comparison(positions, a, d)
        )
        if actual_value != required_value:
            return False
    return True


# ===========================================================================
# 5. EXHAUST ALL 8! RANKINGS, LEAVING EXACTLY THE 12 ELIGIBLE RANKINGS
# ===========================================================================

def enumerate_eligible_rankings(unanimities, tight_triangles):
    """Exhaustively construct the candidate set R(h).

    Key logical point: these are NECESSARY, not sufficient, conditions
    for being a ranking in a four-voter realization.  If a realization
    existed, each of its four voters would definitely be in this set.

    permutations(ALTERNATIVES) enumerates all 8! distinct rankings.
    We check ALL of them, not a random sample or a selected subset.

    To make stage counts auditable, we record:
      rankings_checked       = number of permutations visited (8!)
      respecting_unanimities = number passing the first necessary filter
      eligible_rankings      = ALL rankings passing both filters

    The return value eligible_rankings is a list of pairs:
        (ranking, its_28_coordinate_comparison_vector).
    """
    rankings_checked = 0
    respecting_unanimities = 0
    eligible_rankings = []

    # The itertools permutation generator visits every ordering exactly once.
    for ranking in permutations(ALTERNATIVES):
        rankings_checked += 1
        positions = positions_in_ranking(ranking)

        if not respects_every_unanimity(positions, unanimities):
            continue
        respecting_unanimities += 1

        if not respects_every_tight_triangle(positions, tight_triangles):
            continue

        # This ranking is compatible with every necessary condition.
        # Store its full comparison vector for the final four-voter search.
        eligible_rankings.append((ranking, comparison_vector(positions)))

    # Fail loudly if the supposedly exhaustive enumeration is incomplete.
    require(rankings_checked == factorial(NUMBER_OF_ALTERNATIVES) == 40320,
            "Not all 8! rankings were visited")
    require(len(eligible_rankings) == 12,
            f"Expected 12 eligible rankings, got {len(eligible_rankings)}")

    # Every fractional-support ranking MUST pass the necessary conditions,
    # because its fractional weight is positive.  Also check that the
    # complete enumerated set matches the twelve rankings in the manuscript.
    eligible_order_set = {ranking for ranking, _ in eligible_rankings}
    for fractional_ranking in FRACTIONAL_CERTIFICATE_RANKINGS:
        require(fractional_ranking in eligible_order_set,
                "A supplied fractional-support ranking was excluded")

    displayed_order_set = set(FRACTIONAL_CERTIFICATE_RANKINGS +
                              OTHER_ELIGIBLE_RANKINGS)
    require(eligible_order_set == displayed_order_set,
            "The complete eligible set disagrees with the manuscript")

    return rankings_checked, respecting_unanimities, eligible_rankings


# ===========================================================================
# 6. EXHAUST ALL 1,365 FOUR-VOTER PROFILES FROM THE ELIGIBLE RANKINGS
# ===========================================================================

def exclude_every_four_ranking_profile(eligible_rankings, target_vector):
    """Prove that no multiset of four eligible ranking vectors sums to h.

    Let r be the number of eligible rankings (we verified r=12).
    An ordinary four-voter profile consists of four rankings,
    where the same ranking MAY occur more than once.

    The order of the voters does not affect their pairwise counts.
    Thus, instead of examining all ordered quadruples of rankings, we
    examine every MULTISET of four rankings, with repetitions allowed.

    Number of such multisets:
        C(r+4-1, 4) = C(15,4) = 1,365.

    combinations_with_replacement(range(r), 4) generates every
    nondecreasing quadruple of ranking INDICES (first, second, third, fourth).
    It includes tuples such as (0,0,0,0) and (0,0,0,1), so we are NOT
    accidentally excluding repeated voters.

    For each multiset, add its four 28-coordinate comparison vectors
    and check whether the result equals the target h in EVERY coordinate.

    IMPORTANT: We DO NOT merely rely on the fact that a search failed;
    we count and check every possible multiset, and reject the proof
    immediately if even one exactly realizes h.
    """
    number_of_eligible_rankings = len(eligible_rankings)
    number_of_coordinates = len(UNORDERED_PAIRS)

    # Extract just the comparison vectors; the rankings themselves
    # are retained separately to print a counter-witness if needed.
    vectors = [vector for ranking, vector in eligible_rankings]

    number_of_profiles_checked = 0
    number_of_solutions = 0

    for first_index, second_index, third_index, fourth_index in (
            combinations_with_replacement(range(number_of_eligible_rankings),
                                          NUMBER_OF_VOTERS)):
        number_of_profiles_checked += 1

        first_vector = vectors[first_index]
        second_vector = vectors[second_index]
        third_vector = vectors[third_index]
        fourth_vector = vectors[fourth_index]

        # 'all' checks the coordinate equality for all 28 pairs a<b.
        # It may stop early if a coordinate fails; this is safe,
        # because a successful realization must pass EVERY coordinate.
        realizes_target = all(
            first_vector[coordinate_index]
            + second_vector[coordinate_index]
            + third_vector[coordinate_index]
            + fourth_vector[coordinate_index]
            == target_vector[coordinate_index]
            for coordinate_index in range(number_of_coordinates)
        )

        if realizes_target:
            number_of_solutions += 1
            # If the proposed counterexample were false, make the
            # exact realizing profile visible and FAIL immediately.
            witness = (
                eligible_rankings[first_index][0],
                eligible_rankings[second_index][0],
                eligible_rankings[third_index][0],
                eligible_rankings[fourth_index][0],
            )
            raise RuntimeError(
                "VERIFICATION FAILED: found a genuine four-voter "
                f"realization: {witness}"
            )

    # Check the count of enumerated multisets independently against
    # the elementary combinatorial formula.
    expected_number = comb(number_of_eligible_rankings + NUMBER_OF_VOTERS - 1,
                           NUMBER_OF_VOTERS)
    require(number_of_profiles_checked == expected_number == 1365,
            "The final multiset enumeration was incomplete")
    require(number_of_solutions == 0,
            "A four-ranking realization was found")

    return number_of_profiles_checked, number_of_solutions


# ===========================================================================
# 7. VERIFY THE EXTRA FINITE FACT USED FOR EVERY DEGREE k >= 5
# ===========================================================================

def verify_higher_degree_extension(eligible_rankings):
    """Check the two-comparison fact used in the higher-degree proof.

    The paper proves the extension mathematically, not by testing all k.
    We only need to verify the finite assertion that Q is the UNIQUE
    eligible ranking putting 3 before 1 and 1 before 5.

    For h_k = h+(k-4)*x^Q:
      - exactly H_13 = 1 voter puts 1 before 3 (since Q puts 3 first);
      - exactly 4-H_15 = 3 voters put 5 before 1 (since Q puts 1 first).

    Every other eligible ranking violates at least one of Q's two
    comparisons and so belongs to a set of at most 1+3 = 4 voters.
    The remaining >=k-4 rankings must equal Q.  Cancelling them
    would give a four-ranking realization of h, already excluded.
    """
    q = DISTINGUISHED_RANKING_Q
    q_positions = positions_in_ranking(q)
    require(q_positions[3] < q_positions[1] < q_positions[5],
            "Q does not place 3 before 1 before 5")

    # Check Q is compatible with every saturated constraint, rather than
    # silently relying on its appearance in the fractional certificate.
    unanimous_requirements = unanimous_comparison_requirements()
    triangle_requirements = tight_triangle_requirements()
    require(respects_every_unanimity(q_positions, unanimous_requirements),
            "Q violates an original unanimous comparison")
    require(respects_every_tight_triangle(q_positions, triangle_requirements),
            "Q violates an original tight triangle equality")

    # The eligible set was obtained by COMPLETE enumeration in Part B.
    eligible_orders_with_both_comparisons = [
        ranking for ranking, _ in eligible_rankings
        if (positions_in_ranking(ranking)[3] < positions_in_ranking(ranking)[1]
            < positions_in_ranking(ranking)[5])
    ]
    require(eligible_orders_with_both_comparisons == [q],
            "Q is not the unique eligible ranking with 3 before 1 before 5")

    require(matrix_entry(1, 3) == 1 and matrix_entry(1, 5) == 1,
            "The crucial H_13=H_15=1 entries are incorrect")
    return q


# ===========================================================================
# 8. TOP-LEVEL PROOF DRIVER AND READABLE OUTPUT
# ===========================================================================

def main():
    """Run the mathematical verification in the same order as the paper."""
    argument_parser = argparse.ArgumentParser(
        description="Verify the (n=8,k=4) counterexample by complete enumeration."
    )
    argument_parser.add_argument(
        "--show-eligible", action="store_true",
        help="Print all 12 rankings surviving the necessary-condition filters."
    )
    argument_parser.add_argument(
        "--show-tight", action="store_true",
        help="Print each tight triangle and its required per-voter value."
    )

    # Unlike the attached 9,3 script, do not hard-code option values here:
    # parse_known_args() honors command-line flags and also tolerates
    # unrelated arguments inserted by notebooks such as Jupyter/Colab.
    options, _ = argument_parser.parse_known_args()

    start_time = perf_counter()  # Wall-clock duration of this single run.

    print("Verification of the n=8, k=4 counterexample")
    print("=" * 52)

    # First validate the matrix and the explicit fractional-support orders.
    validate_input_data()
    target_vector = target_comparison_vector()
    print(f"Target dimension: {len(target_vector)} comparison coordinates")

    # Part A: positive certificate of fractional realizability.
    verified_coordinates = verify_fractional_realization(target_vector)
    print("\nPART A: FRACTIONAL REALIZABILITY")
    print("  Eight explicitly supplied rankings, each of weight 1/2.")
    print(f"  Exact equality sum(x^pi) = 2*h checked in {verified_coordinates} coordinates.")
    print("  Total fractional weight: 4")

    # Part B, first stage: identify the filters required of every voter.
    unanimities = unanimous_comparison_requirements()
    tight_triangles = tight_triangle_requirements()
    require(len(unanimities) == 5,
            f"Expected five unanimous comparisons, got {len(unanimities)}")
    require(len(tight_triangles) == 14,
            f"Expected 14 tight triangles, got {len(tight_triangles)}")

    print("\nPART B: EXCLUDE FOUR ORDINARY VOTERS")
    print(f"  Unanimous comparisons: {len(unanimities)}")
    for a, b, required_bit in unanimities:
        # The bit 1 indicates a before b; bit 0 indicates b before a.
        winner, loser = (a, b) if required_bit else (b, a)
        print(f"    Every voter must rank {winner} before {loser}.")
    print(f"  Tight triangle equalities: {len(tight_triangles)}")

    if options.show_tight:
        for a, b, d, forced_value in tight_triangles:
            print(f"    Triangle ({a},{b},{d}): "
                  f"x_{a}{b} + x_{b}{d} - x_{a}{d} = {forced_value}")

    # Enumerate the complete set R(h) of eligible rankings.
    total_rankings, passed_unanimities, eligible_rankings = (
        enumerate_eligible_rankings(unanimities, tight_triangles)
    )
    print(f"  All rankings examined: {total_rankings:,}")
    print(f"  Rankings respecting unanimities: {passed_unanimities:,}")
    print(f"  Rankings also respecting tight triangles: {len(eligible_rankings)}")

    if options.show_eligible:
        print("\n  Complete list of the 12 eligible rankings:")
        for rank_number, (ranking, _) in enumerate(eligible_rankings, start=1):
            print(f"    {rank_number:2d}: {' '.join(map(str, ranking))}")

    # Exhaust ALL four-voter multisets among those eligible rankings.
    checked_profiles, found_solutions = exclude_every_four_ranking_profile(
        eligible_rankings, target_vector
    )
    print(f"  Four-ranking multisets checked: {checked_profiles:,}")
    print(f"  Exact four-ranking realizations found: {found_solutions}")

    # Part C: just the extra finite condition used by the analytic proof
    # of the extension to every higher degree.
    q = verify_higher_degree_extension(eligible_rankings)
    print("\nPART C: FINITE CHECK FOR THE k>=5 EXTENSION")
    print("  Exactly one eligible ranking places 3 before 1 before 5.")
    print(f"  That ranking is Q = {q}")
    print("  H_13=H_15=1: at most 1+3=4 voters can differ from Q.")

    elapsed_time = perf_counter() - start_time
    print("\nSUCCESS: h belongs to 4*LO_8 but has no four-ranking realization.")
    print("The counting argument therefore applies to every k>=5.")
    print(f"Elapsed time: {elapsed_time:.3f} seconds")


if __name__ == "__main__":
    main()
