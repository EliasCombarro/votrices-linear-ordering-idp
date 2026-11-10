#!/usr/bin/env python3
"""Verify the nine-alternative, three-voter counterexample.

MATHEMATICAL OBJECTS
--------------------
We have nine alternatives, labeled 1, 2, ..., 9.  A ranking is a permutation
of these nine labels, listed from first (preferred) to last.

For a ranking pi, its pairwise-comparison vector x^pi has one coordinate for
each pair a<b.  That coordinate is:

    x^pi_ab = 1 if a occurs BEFORE b in pi,
              0 if b occurs BEFORE a in pi.

There are C(9,2) = 36 coordinates, one for each unordered pair.

The matrix M below encodes an INTEGER target vector c using M_ab = c_ab
for a<b and the convention M_ab + M_ba = 3 for a != b.
The claim is that c belongs to 3*LO_9 but cannot be written as the sum
of three ranking vectors.

WHAT THE PROGRAM PROVES
-----------------------
(A) FRACTIONAL REALIZABILITY: The sum of the comparison vectors of six
    explicitly given rankings is exactly 2*c.  Giving each ranking
    weight 1/2 therefore realizes c with total weight 6*(1/2) = 3.

(B) NO THREE-VOTER REALIZATION:
    1.  Compute the unanimous comparisons of c.  Every ranking used
        in a three-voter realization must respect them.
    2.  Compute all tight triangle inequalities of c.  Every ranking
        used in a three-voter realization must attain the same bound.
    3.  Enumerate ALL 9! = 362,880 rankings, keeping only those that
        satisfy the restrictions in steps 1 and 2.  Exactly 33 survive.
    4.  Enumerate ALL C(33+3-1,3) = 6,545 multisets of THREE rankings
        from those 33.  Repeated rankings ARE allowed.  None has sum c.

WHY THIS IS A COMPLETE PROOF
----------------------------
Any three-voter realization is a multiset of three total rankings.
Every one of those rankings MUST satisfy the unanimous and tight-triangle
conditions; thus all three belong to our explicitly enumerated 33.
Step B4 checks every possible multiset of those 33, and hence excludes
all possible three-voter realizations.  This is a *necessary-condition*
filter followed by a *complete finite search*, not an assumption that
unanimity and triangle inequalities characterize 3*LO_9.

USAGE
-----
    python3 verify_n9_k3_simple.py

For more detail about the intermediate filters:
    python3 verify_n9_k3_simple.py --show-eligible --show-tight

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

# We are studying rankings of nine alternatives with three ordinary voters.
NUMBER_OF_ALTERNATIVES = 9
NUMBER_OF_VOTERS = 3

# This is the matrix M displayed in Section 4 of the manuscript.
#
# PAIRWISE_COUNTS[a-1][b-1] = M_ab, the desired number of voters who
# prefer alternative a to alternative b, where a,b are labels in 1..9.
#
# - Python array indices start at 0; mathematical labels start at 1.
# - Every diagonal entry is 0 because we do not compare an alternative
#   against itself.
# - For a != b, M_ab + M_ba must equal NUMBER_OF_VOTERS = 3.
# - The upper-triangular entries, in lexicographic order, form the
#   36-dimensional target vector c.
PAIRWISE_COUNTS = (
    (0, 1, 1, 1, 1, 1, 2, 0, 0),
    (2, 0, 1, 1, 2, 1, 1, 2, 1),
    (2, 2, 0, 1, 1, 2, 1, 1, 1),
    (2, 2, 2, 0, 1, 1, 1, 2, 0),
    (2, 1, 2, 2, 0, 2, 1, 1, 1),
    (2, 2, 1, 2, 1, 0, 1, 1, 2),
    (1, 2, 2, 2, 2, 2, 0, 1, 1),
    (3, 1, 2, 1, 2, 2, 2, 0, 1),
    (3, 2, 2, 3, 2, 1, 2, 2, 0),
)

# These six rankings are the EXPLICIT FRACTIONAL CERTIFICATE.
#
# Each row lists all alternatives from FIRST to LAST.  Each has weight 1/2.
# For example, (7,3,6,...) means 7 precedes 3, and 3 precedes 6.
# Their six comparison vectors should sum to 2*c, coordinate by coordinate.
FRACTIONAL_CERTIFICATE_RANKINGS = (
    (7, 3, 6, 2, 9, 5, 4, 8, 1),
    (7, 6, 2, 9, 5, 4, 8, 3, 1),
    (8, 5, 3, 6, 9, 1, 7, 4, 2),
    (8, 5, 6, 9, 1, 7, 4, 3, 2),
    (9, 4, 2, 8, 1, 7, 5, 3, 6),
    (9, 4, 3, 2, 8, 1, 7, 5, 6),
)

# Mathematical alternative labels are 1,2,...,9 (NOT 0,1,...,8).
ALTERNATIVES = tuple(range(1, NUMBER_OF_ALTERNATIVES + 1))

# All 36 unordered pairs (a,b), always with a<b, in lexicographic order.
# This is also the exact coordinate order of our target and ranking vectors.
UNORDERED_PAIRS = tuple(combinations(ALTERNATIVES, 2))

# All 84 unordered triples (a,b,d), always with a<b<d.
# The variable name 'd' is used for the third alternative to avoid
# confusing it with the target vector c.
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
    """Return M_ab using the mathematical labels a,b in 1..9.

    For example, matrix_entry(1,8) = 0 means that in the target matrix,
    alternative 1 precedes alternative 8 with count zero.
    """
    return PAIRWISE_COUNTS[first_alternative - 1][second_alternative - 1]


def positions_in_ranking(ranking):
    """Return each alternative's POSITION in the ranking.

    If ranking = (8,1,2,...) then result[8]=0, result[1]=1,
    result[2]=2.  The entry at position zero is unused, allowing us
    to index by the mathematical labels 1..9 without subtracting one.

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
    """Return all 36 upper-triangular comparison coordinates of a ranking.

    The result is a tuple of zeros and ones, ordered exactly as
    UNORDERED_PAIRS = ((1,2),(1,3),...,(8,9)).

    To compare a sum of ranking vectors with the target, we only need
    these 36 coordinates; opposite-direction coordinates are redundant.
    """
    return tuple(
        ranking_comparison(positions, a, b)
        for a, b in UNORDERED_PAIRS
    )


def validate_input_data():
    """Validate the entire target matrix and the six explicit rankings.

    These checks protect against typos in the manuscript or transcription
    errors in this file before any mathematical conclusion is attempted.
    """
    require(len(PAIRWISE_COUNTS) == NUMBER_OF_ALTERNATIVES,
            "The matrix does not have nine rows")

    for row_number, matrix_row in enumerate(PAIRWISE_COUNTS, start=1):
        require(len(matrix_row) == NUMBER_OF_ALTERNATIVES,
                f"Matrix row {row_number} does not have nine columns")

    for a in ALTERNATIVES:
        require(matrix_entry(a, a) == 0,
                f"Diagonal matrix entry M_{a}{a} is not zero")

    for a, b in UNORDERED_PAIRS:
        forward_count = matrix_entry(a, b)
        reverse_count = matrix_entry(b, a)
        require(isinstance(forward_count, int) and
                isinstance(reverse_count, int),
                f"Non-integer matrix entry for pair ({a},{b})")
        require(0 <= forward_count <= NUMBER_OF_VOTERS and
                0 <= reverse_count <= NUMBER_OF_VOTERS,
                f"Entry outside 0,...,3 for pair ({a},{b})")
        require(forward_count + reverse_count == NUMBER_OF_VOTERS,
                f"M_{a}{b}+M_{b}{a} is not three")

    require(len(UNORDERED_PAIRS) == comb(NUMBER_OF_ALTERNATIVES, 2) == 36,
            "Incorrect 36-coordinate indexing")
    require(len(UNORDERED_TRIPLES) == comb(NUMBER_OF_ALTERNATIVES, 3) == 84,
            "Incorrect 84-triple indexing")

    require(len(FRACTIONAL_CERTIFICATE_RANKINGS) == 6,
            "The fractional certificate must contain six rankings")

    sorted_labels = list(ALTERNATIVES)
    for certificate_index, ranking in enumerate(
            FRACTIONAL_CERTIFICATE_RANKINGS, start=1):
        require(sorted(ranking) == sorted_labels,
                f"Fractional ranking {certificate_index} is not a permutation")


def target_comparison_vector():
    """Construct c using M_ab for a<b, in the 36-coordinate order."""
    return tuple(matrix_entry(a, b) for a, b in UNORDERED_PAIRS)


# ===========================================================================
# 3. VERIFY FRACTIONAL REALIZABILITY 
# ===========================================================================

def verify_fractional_realization(target_vector):
    """Check sum_{six rankings pi} x^pi = 2*c EXACTLY.

    Every ranking has fractional weight 1/2.  Therefore:

        (1/2) * sum_{six pi} x^pi = c,
        total fractional weight = 6 * (1/2) = 3.

    We avoid floating point: checking the equivalent integer equality
    sum x^pi = 2*c is exact, with no rounding or solver tolerances.
    """
    number_of_coordinates = len(UNORDERED_PAIRS)  # Should be 36.

    # Sum of the comparison vectors across the SIX supplied rankings.
    summed_comparison_counts = [0] * number_of_coordinates

    for ranking in FRACTIONAL_CERTIFICATE_RANKINGS:
        positions = positions_in_ranking(ranking)
        ranking_vector = comparison_vector(positions)

        for coordinate_index, comparison_bit in enumerate(ranking_vector):
            summed_comparison_counts[coordinate_index] += comparison_bit

    # Explicitly compare every one of the 36 coordinates with twice c.
    for coordinate_index, (a, b) in enumerate(UNORDERED_PAIRS):
        actual_sum = summed_comparison_counts[coordinate_index]
        required_sum = 2 * target_vector[coordinate_index]
        require(actual_sum == required_sum,
                f"Fractional realization fails for pair ({a},{b}): "
                f"actual {actual_sum}, required {required_sum}")

    # Six weights each equal 1/2; multiply by 2 to avoid fractions.
    require(len(FRACTIONAL_CERTIFICATE_RANKINGS) == 2 * NUMBER_OF_VOTERS,
            "The certificate does not have total fractional weight three")

    return number_of_coordinates


# ===========================================================================
# 4. EXTRACT THE NECESSARY RESTRICTIONS ON A HYPOTHETICAL VOTER
# ===========================================================================

def unanimous_comparison_requirements():
    """List comparisons that EVERY voter in a realization must obey.

    If M_ab = 3 for a<b, all three voters must put a before b,
    so each ranking must have x^pi_ab = 1.

    If M_ab = 0, all three voters must put b before a,
    so each ranking must have x^pi_ab = 0.

    Values M_ab=1 or 2 impose NO orientation on an individual voter.
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

    For the target c, define the integer

        target_triangle = M_ab + M_bd - M_ad.

    If c is the sum of THREE ranking vectors, target_triangle is
    the sum of three separate values, each in {0,1}.  Thus it must
    lie between 0 and 3.  Moreover:

      * If target_triangle = 0, EVERY voter must contribute 0.
      * If target_triangle = 3, EVERY voter must contribute 1.
      * If target_triangle = 1 or 2, this triple alone does not
        force the same contribution from all voters.

    We call a triangle "tight" precisely when target_triangle
    equals 0 or 3.  These are NECESSARY constraints on each voter.

    WORKED EXAMPLE, matching the manuscript (alternatives 1,2,8):

        M_12=1, M_28=2, M_18=0, so
        M_12 + M_28 - M_18 = 1+2-0 = 3.

    Therefore EVERY hypothetical voter must satisfy

        x^pi_12 + x^pi_28 - x^pi_18 = 1.

    Unanimity already forces 8 ahead of 1, meaning x^pi_18=0.
    Hence x^pi_12 + x^pi_28 = 1.

    With 8 ahead of 1, the two possible relative orders are
    8-1-2 and 2-8-1; the order 8-2-1 would make both
    x^pi_12 and x^pi_28 zero, violating the tight equality.

    Output entries are (a, b, d, forced_individual_triangle_value).
    """
    requirements = []

    for a, b, d in UNORDERED_TRIPLES:
        target_triangle = (
            matrix_entry(a, b)
            + matrix_entry(b, d)
            - matrix_entry(a, d)
        )

        # We also check that c satisfies all 84 triangle inequalities.
        # If it did not, c could not be fractionally realizable.
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
# 5. EXHAUST ALL 9! RANKINGS, LEAVING EXACTLY THE 33 ELIGIBLE RANKINGS
# ===========================================================================

def enumerate_eligible_rankings(unanimities, tight_triangles):
    """Exhaustively construct the candidate set R(c).

    Key logical point: these are NECESSARY, not sufficient, conditions
    for being a ranking in a three-voter realization.  If a realization
    existed, each of its three voters would definitely be in this set.

    permutations(ALTERNATIVES) enumerates all 9! distinct rankings.
    We check ALL of them, not a random sample or a selected subset.

    To make stage counts auditable, we record:
      rankings_checked       = number of permutations visited (9!)
      respecting_unanimities = number passing the first necessary filter
      eligible_rankings      = ALL rankings passing both filters

    The return value eligible_rankings is a list of pairs:
        (ranking, its_36_coordinate_comparison_vector).
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
        # Store its full comparison vector for the final triple search.
        eligible_rankings.append((ranking, comparison_vector(positions)))

    # Fail loudly if the supposedly exhaustive enumeration is incomplete.
    require(rankings_checked == factorial(NUMBER_OF_ALTERNATIVES),
            "Not all 9! rankings were visited")
    require(len(eligible_rankings) == 33,
            f"Expected 33 eligible rankings, got {len(eligible_rankings)}")

    # A second consistency test: all six explicit fractional-support orders
    # MUST pass the necessary conditions, because their weights are positive.
    eligible_order_set = {ranking for ranking, _ in eligible_rankings}
    for fractional_ranking in FRACTIONAL_CERTIFICATE_RANKINGS:
        require(fractional_ranking in eligible_order_set,
                "A supplied fractional-support ranking was excluded")

    return rankings_checked, respecting_unanimities, eligible_rankings


# ===========================================================================
# 6. EXHAUST ALL 6,545 THREE-VOTER PROFILES FROM THE ELIGIBLE RANKINGS
# ===========================================================================

def exclude_every_three_ranking_profile(eligible_rankings, target_vector):
    """Prove that no triple of eligible ranking vectors sums to c.

    Let r be the number of eligible rankings (we verified r=33).
    An ordinary three-voter profile consists of three rankings,
    where the same ranking MAY occur more than once.

    The order of the voters does not affect their pairwise counts.
    Thus, instead of examining all ordered triples of rankings, we
    examine every MULTISET of three rankings, with repetitions allowed.

    Number of such multisets:
        C(r+3-1, 3) = C(35,3) = 6,545.

    combinations_with_replacement(range(r), 3) generates every
    nondecreasing triple of ranking INDICES (first, second, third).
    It includes triples such as (0,0,0) and (0,0,1), so we are NOT
    accidentally excluding repeated voters.

    For each multiset, add its three 36-coordinate comparison vectors
    and check whether the result equals the target c in EVERY coordinate.

    IMPORTANT: We DO NOT merely rely on the fact that a search failed;
    we count and check every possible multiset, and reject the proof
    immediately if even one exactly realizes c.
    """
    number_of_eligible_rankings = len(eligible_rankings)
    number_of_coordinates = len(UNORDERED_PAIRS)

    # Extract just the comparison vectors; the rankings themselves
    # are retained separately to print a counter-witness if needed.
    vectors = [vector for ranking, vector in eligible_rankings]

    number_of_profiles_checked = 0
    number_of_solutions = 0

    for first_index, second_index, third_index in combinations_with_replacement(
            range(number_of_eligible_rankings), NUMBER_OF_VOTERS):
        number_of_profiles_checked += 1

        first_vector = vectors[first_index]
        second_vector = vectors[second_index]
        third_vector = vectors[third_index]

        # 'all' checks the coordinate equality for all 36 pairs a<b.
        # It may stop early if a coordinate fails; this is safe,
        # because a successful realization must pass EVERY coordinate.
        realizes_target = all(
            first_vector[coordinate_index]
            + second_vector[coordinate_index]
            + third_vector[coordinate_index]
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
            )
            raise RuntimeError(
                "VERIFICATION FAILED: found a genuine three-voter "
                f"realization: {witness}"
            )

    # Check the count of enumerated multisets independently against
    # the elementary combinatorial formula.
    expected_number = comb(number_of_eligible_rankings + NUMBER_OF_VOTERS - 1,
                           NUMBER_OF_VOTERS)
    require(number_of_profiles_checked == expected_number == 6545,
            "The final multiset enumeration was incomplete")
    require(number_of_solutions == 0,
            "A three-ranking realization was found")

    return number_of_profiles_checked, number_of_solutions


# ===========================================================================
# 7. TOP-LEVEL PROOF DRIVER AND READABLE OUTPUT
# ===========================================================================

def main():
    """Run the mathematical verification in the same order as the paper."""
    argument_parser = argparse.ArgumentParser(
        description="Verify the (n=9,k=3) counterexample by complete enumeration."
    )
    argument_parser.add_argument(
        "--show-eligible", action="store_true",
        help="Print all 33 rankings surviving the necessary-condition filters."
    )
    argument_parser.add_argument(
        "--show-tight", action="store_true",
        help="Print each tight triangle and its required per-voter value."
    )
    options = argument_parser.parse_args(args=["--show-eligible", "--show-tight"])

    start_time = perf_counter()  # Wall-clock duration of this single run.

    print("Verification of the n=9, k=3 counterexample")
    print("=" * 52)

    # First validate the matrix and the explicit fractional-support orders.
    validate_input_data()
    target_vector = target_comparison_vector()
    print(f"Target dimension: {len(target_vector)} comparison coordinates")

    # Part A: positive certificate of fractional realizability.
    verified_coordinates = verify_fractional_realization(target_vector)
    print("\nPART A: FRACTIONAL REALIZABILITY")
    print("  Six explicitly supplied rankings, each of weight 1/2.")
    print(f"  Exact equality sum(x^pi) = 2*c checked in {verified_coordinates} coordinates.")
    print("  Total fractional weight: 3")

    # Part B, first stage: identify the filters required of every voter.
    unanimities = unanimous_comparison_requirements()
    tight_triangles = tight_triangle_requirements()
    require(len(unanimities) == 3,
            f"Expected three unanimous comparisons, got {len(unanimities)}")
    require(len(tight_triangles) == 28,
            f"Expected 28 tight triangles, got {len(tight_triangles)}")

    print("\nPART B: EXCLUDE THREE ORDINARY VOTERS")
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

    # Enumerate the complete set R(c) of eligible rankings.
    total_rankings, passed_unanimities, eligible_rankings = (
        enumerate_eligible_rankings(unanimities, tight_triangles)
    )
    print(f"  All rankings examined: {total_rankings:,}")
    print(f"  Rankings respecting unanimities: {passed_unanimities:,}")
    print(f"  Rankings also respecting tight triangles: {len(eligible_rankings)}")

    if options.show_eligible:
        print("\n  Complete list of the 33 eligible rankings:")
        for rank_number, (ranking, _) in enumerate(eligible_rankings, start=1):
            print(f"    {rank_number:2d}: {' '.join(map(str, ranking))}")

    # Exhaust ALL three-voter multisets among those eligible rankings.
    checked_profiles, found_solutions = exclude_every_three_ranking_profile(
        eligible_rankings, target_vector
    )
    print(f"  Three-ranking multisets checked: {checked_profiles:,}")
    print(f"  Exact three-ranking realizations found: {found_solutions}")

    elapsed_time = perf_counter() - start_time
    print("\nSUCCESS: c belongs to 3*LO_9 but has no three-ranking realization.")
    print(f"Elapsed time: {elapsed_time:.3f} seconds")


if __name__ == "__main__":
    main()
