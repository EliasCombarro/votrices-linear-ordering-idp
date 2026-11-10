#include <algorithm>   // min, next_permutation
#include <cstdint>     // uint32_t and UINT32_MAX
#include <fstream>     // ifstream
#include <iostream>    // cout, endl
#include <set>         // set: stores distinct canonical encodings
#include <stdexcept>   // runtime_error
#include <string>      // string
#include <vector>      // vector

using namespace std;

/*
 * Independently generate all tournaments up to isomorphism on at most
 * eight vertices, then compare the generated eight-vertex classes with
 * a supplied tournament catalogue.
 *
 * Tournament encoding (for vertices 0, ..., n-1):
 *   - There is one bit for every unordered pair {i,j}, listed in the
 *     order (0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2,n-1).
 *   - For i < j, bit 1 means i -> j; bit 0 means j -> i.
 *   - Bit positions are numbered from the RIGHT, starting at 0.
 *
 * There are n*(n-1)/2 bits. For n <= 8, at most 28 bits are needed,
 * so the whole tournament fits into an unsigned 32-bit integer.
 *
 * To avoid counting isomorphic tournaments more than once, we compute
 * a canonical encoding: the smallest encoding obtained by relabeling
 * the vertices while placing outdegree classes in increasing order.
 * This is safe because every tournament isomorphism preserves outdegree.
 */

// Return the same canonical integer for two tournaments if and only if
// they are isomorphic. The input code is the bit encoding described above.
uint32_t canonical(uint32_t code, int n) {
    // a[i][j] is true exactly when the directed edge i -> j exists.
    // {} initializes every entry to false (and each score to zero).
    bool a[8][8]{};
    int score[8]{};

    // t is the bit position corresponding to the current pair (i,j).
    int t = 0;

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++, t++) {
            // Shift code right by t positions so that bit t becomes
            // the rightmost bit. AND with 1 discards all other bits.
            // The result is 1 if i -> j, and 0 if j -> i.
            // Example: code = 5 = binary 101; t = 1 gives (5 >> 1) & 1 = 0.
            a[i][j] = (code >> t) & 1;

            // A tournament has exactly one direction for each pair.
            a[j][i] = !a[i][j];

            // Conditional operator: condition ? if_true : if_false.
            // If a[i][j] is true, the source of the edge is i;
            // otherwise, the source is j. Increment its outdegree.
            // Equivalent to:
            //     if (a[i][j]) score[i]++;
            //     else         score[j]++;
            score[a[i][j] ? i : j]++;
        }
    }

    // Partition vertices into groups ("cells") of equal outdegree.
    // The groups appear in INCREASING order of outdegree because d
    // increases from 0 to n-1. Empty groups are simply omitted.
    // We need the groups, not a second copy of their degree values:
    // the degrees are still available in the score[] array.
    vector<vector<int>> cells;

    for (int d = 0; d < n; d++) {
        vector<int> c;  // All vertices of outdegree d.

        for (int i = 0; i < n; i++) {
            if (score[i] == d) {
                c.push_back(i);  // Append vertex i to this degree class.
            }
        }

        if (!c.empty()) {
            cells.push_back(c);  // Keep only nonempty degree classes.
        }
    }

    // best holds the smallest relabeled encoding found so far.
    // UINT32_MAX is the largest possible uint32_t value (2^32 - 1),
    // so the first legitimate 28-bit encoding will be smaller.
    uint32_t best = UINT32_MAX;

    // p is a partially constructed relabeling: p[r] is the ORIGINAL
    // vertex that will occupy position r in the NEW vertex ordering.
    vector<int> p;

    // Recursive lambda for enumerating every ordering within each cell.
    // [&] captures nearby variables (cells, p, a, best, n) by reference.
    // k is the index of the next outdegree class to append to p.
    // "self" is passed explicitly so that this lambda can call itself.
    auto visit = [&](auto&& self, int k) -> void {
        // Base case: all degree classes have been processed; p now
        // lists every vertex exactly once, giving a complete relabeling.
        if (k == (int)cells.size()) {
            uint32_t z = 0;  // Encoding of the tournament under p.
            int e = 0;       // Bit position corresponding to (i,j).

            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++, e++) {
                    // Positions i,j in the NEW ordering correspond to
                    // original vertices p[i],p[j]. Check that edge.
                    if (a[p[i]][p[j]]) {
                        // 1u << e has only bit e set. The |= operator
                        // sets that bit of z without changing other bits.
                        z |= 1u << e;
                    }
                }
            }

            // Keep the numerically smallest encoding among all allowed
            // relabelings. This makes the representation canonical.
            best = min(best, z);
            return;  // This complete permutation has been processed.
        }

        // Make a COPY of the next outdegree class. Its vertices are
        // initially sorted, as they were collected in vertex order.
        auto c = cells[k];

        do {
            // Append this ordering of the current class to p.
            for (int v : c) {
                p.push_back(v);
            }

            // Recursively explore every ordering of the later classes.
            self(self, k + 1);

            // BACKTRACK: remove this class from p, restoring its state
            // before the append. Now the next ordering can be tried.
            p.resize(p.size() - c.size());

            // Generate the next lexicographic permutation of c.
            // Since c began sorted, the do/while visits every ordering
            // exactly once, stopping when next_permutation returns false.
        } while (next_permutation(c.begin(), c.end()));
    };

    // Begin with the first (lowest-outdegree) class and return the
    // smallest encoding found after exploring all class permutations.
    visit(visit, 0);
    return best;
}

int main(int argc, char** argv) {
    // The sole command-line argument is the path to McKay's tournament
    // catalogue, with one 28-character binary encoding per line.
    if (argc != 2) {
        throw runtime_error("supply archive catalogue path");
    }

    // For n=1 there is one tournament, with no edges, encoded by zero.
    // A set automatically discards duplicate canonical encodings.
    set<uint32_t> generated{0};

    // Known counts give a quick sanity check at each size. They are NOT
    // the proof of completeness: completeness follows inductively from
    // generating every possible way to attach one new vertex.
    int expected[] = {0, 1, 1, 2, 4, 12, 56, 456, 6880};

    cout << "{\"independently_generated_counts\":[1";

    for (int n = 2; n <= 8; n++) {
        // We have all nonisomorphic (n-1)-vertex tournaments in
        // generated; construct all their n-vertex extensions in next.
        set<uint32_t> next;

        for (uint32_t parent : generated) {
            // A newly added vertex is numbered n-1. Each of the n-1
            // edges connecting it to old vertices can point either way.
            // Hence there are 2^(n-1) possible bit patterns (joins).
            for (int joins = 0; joins < (1 << (n - 1)); joins++) {
                uint32_t child = 0;
                int dst = 0;  // Bit index in the new n-vertex encoding.
                int src = 0;  // Bit index in the old (n-1)-vertex encoding.

                for (int i = 0; i < n; i++) {
                    for (int j = i + 1; j < n; j++, dst++) {
                        // If j is the newly added vertex n-1, take the
                        // orientation i -> (n-1) from bit i of joins.
                        // Otherwise, copy the edge orientation from
                        // the parent tournament, using its next bit.
                        bool forward = (j == n - 1)
                            ? ((joins >> i) & 1)
                            : ((parent >> src++) & 1);

                        if (forward) {
                            child |= 1u << dst;  // Set edge i -> j.
                        }
                    }
                }

                // Canonicalization merges isomorphic extensions;
                // set insertion also removes all duplicate encodings.
                next.insert(canonical(child, n));
            }
        }

        // Every n-vertex tournament arises from deleting one vertex,
        // so extensions of the complete (n-1)-vertex class list cover
        // every n-vertex class. Retain exactly one code per class.
        generated = move(next);

        if (generated.size() != static_cast<size_t>(expected[n])) {
            throw runtime_error("unexpected independently generated count");
        }
        cout << "," << generated.size();
    }

    cout << "],";

    // Read the supplied eight-vertex catalogue only AFTER independently
    // generating the classes. A missing file causes an explicit failure.
    ifstream in(argv[1]);
    if (!in) {
        throw runtime_error("missing catalogue");
    }

    set<uint32_t> supplied;
    string line;
    int lines = 0;

    while (getline(in, line)) {
        // Each n=8 tournament must be exactly 28 bits of 0s and 1s.
        if (line.size() != 28 ||
            line.find_first_not_of("01") != string::npos) {
            throw runtime_error("invalid tournament encoding");
        }

        // Convert this textual binary encoding into the integer
        // convention used by canonical(): character e is bit e.
        uint32_t z = 0;
        for (int e = 0; e < 28; e++) {
            if (line[e] == '1') {
                z |= 1u << e;
            }
        }

        // Canonicalize each catalogue entry and check that none are
        // isomorphic. insert(...).second is false for duplicates.
        if (!supplied.insert(canonical(z, 8)).second) {
            throw runtime_error(
                "isomorphic duplicate in supplied catalogue"
            );
        }
        lines++;
    }

    // Stronger than checking that both lists contain 6880 entries:
    // require exact equality of their canonical representative sets.
    if (supplied != generated) {
        throw runtime_error(
            "catalogue fails independent exhaustive generation"
        );
    }

    // The JSON result is printed only after all checks have passed.
    cout << "\"catalogue_entries\":" << lines
         << ",\"complete_up_to_isomorphism\":true}" << endl;
}
