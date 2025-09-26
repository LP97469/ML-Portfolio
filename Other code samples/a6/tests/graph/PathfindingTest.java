package graph;

import graph.Pathfinding.PathEnd;
import java.util.*;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static graph.SimpleGraph.*;

/**
 * Uses the `SimpleGraph` class to verify the functionality of the `Pathfinding` class.
 */
public class PathfindingTest {


    // TODO 12: Complete the definition of the empty test cases so that they match their
    //  `@DisplayName` descriptions. We have provided two complete test cases as examples, as
    //   well as some helper methods for constructing graphs and checking paths that will likely
    //   be useful.


    /*
     * Text graph format ([weight] is optional):
     * Directed edge: startLabel -> endLabel [weight]
     * Undirected edge (so two directed edges in both directions): startLabel -- endLabel [weight]
     */


    // a small, strongly-connected graph consisting of three vertices and four directed edges
    public static final String graph1 = """
           A -> B 2
           A -- C 6
           B -> C 3
           """;


    @DisplayName("WHEN we compute the `pathInfo` from a vertex `v`, THEN it includes an correct "
                 + "entry for each vertex `w` reachable along a non-backtracking path from `v`.")
    @Nested
    class pathInfoTest {


        // Recall that "strongly connected" describes a graph that includes a (directed) path from
        // any vertex to any other vertex
        @DisplayName("In a strongly connected graph with no `previousEdge`.")
        @Test
        void testStronglyConnectedNoPrevious() {
            SimpleGraph g = SimpleGraph.fromText(graph1);
            SimpleVertex va = g.getVertex("A");
            SimpleVertex vb = g.getVertex("B");
            SimpleVertex vc = g.getVertex("C");


            // compute paths from source vertex "A"
            Map<SimpleVertex, PathEnd<SimpleEdge>> paths = Pathfinding.pathInfo(va, null);
            assertEquals(3, paths.size()); // all vertices are reachable
            assertEquals(0, paths.get(va).distance());
            // since the shortest path A -> A is empty, we can't assert anything about its last edge
            assertEquals(2, paths.get(vb).distance());
            assertEquals(g.getEdge(va, vb), paths.get(vb).lastEdge());
            assertEquals(5, paths.get(vc).distance());
            assertEquals(g.getEdge(vb, vc), paths.get(vc).lastEdge());


            // compute paths from source vertex "B"
            paths = Pathfinding.pathInfo(vb, null);
            assertEquals(3, paths.size()); // all vertices are reachable
            assertEquals(9, paths.get(va).distance());
            assertEquals(g.getEdge(vc, va), paths.get(va).lastEdge());
            assertEquals(0, paths.get(vb).distance());
            assertEquals(3, paths.get(vc).distance());
            assertEquals(g.getEdge(vb, vc), paths.get(vc).lastEdge());


            // compute paths from source vertex "C"
            paths = Pathfinding.pathInfo(vc, null);
            assertEquals(3, paths.size()); // all vertices are reachable
            assertEquals(6, paths.get(va).distance());
            assertEquals(g.getEdge(vc, va), paths.get(va).lastEdge());
            assertEquals(8, paths.get(vb).distance());
            assertEquals(g.getEdge(va, vb), paths.get(vb).lastEdge());
            assertEquals(0, paths.get(vc).distance());
        }


        @DisplayName("In a graph that is *not* strongly connected and `pathInfo` is computed "
                     + "starting from a vertex that cannot reach all other vertices.")
        @Test
        void testNotStronglyConnected() {
            SimpleGraph g = SimpleGraph.fromText("B -> A 2");
            SimpleVertex va = g.getVertex("A");
            SimpleVertex vb = g.getVertex("B");


            Map<SimpleVertex, PathEnd<SimpleEdge>> paths = Pathfinding.pathInfo(va, null);
            assertEquals(1, paths.size()); // only va is reachable
            assertTrue(paths.containsKey(va));
            assertFalse(paths.containsKey(vb));
        }


        @DisplayName(
                "In a strongly connected graph with a `previousEdge` that prevents some vertex "
                + "from being reached.")
        @Test
        void testStronglyConnectedPreviousStillReachable() {
            SimpleGraph g = SimpleGraph.fromText("""
                       C -> A 1
                       A -> C 1
                   """);
            SimpleVertex va = g.getVertex("A");
            SimpleVertex vc = g.getVertex("C");


            SimpleEdge previousEdge = g.getEdge(vc, va);


            Map<SimpleVertex, PathEnd<SimpleEdge>> paths = Pathfinding.pathInfo(va, previousEdge);


            assertTrue(paths.containsKey(va));
            assertFalse(paths.containsKey(vc));
        }




        @DisplayName("In a graph where the shortest path with backtracking is shorter than the "
                     + "shortest non-backtracking path.")
        @Test
        void testBacktrackingShorter() {
            SimpleGraph g = SimpleGraph.fromText("""
                   A -> B 1
                   B -> A 1
                   A -> C 10
                   """);
            SimpleVertex va = g.getVertex("A");
            SimpleVertex vb = g.getVertex("B");
            SimpleVertex vc = g.getVertex("C");


            Map<SimpleVertex, PathEnd<SimpleEdge>> paths = Pathfinding.pathInfo(va, null);
            assertEquals(3, paths.size());
            assertEquals(0, paths.get(va).distance());
            assertEquals(1, paths.get(vb).distance());
            assertEquals(10, paths.get(vc)
                    .distance()); // can't take A -> B -> A -> C since B -> A is backtracking
        }


        @DisplayName("In a graph where some shortest path includes at least 3 edges.")
        @Test
        void testLongerPaths() {
            SimpleGraph g = SimpleGraph.fromText("""
                   A -> B 1
                   B -> C 1
                   C -> D 1
                   """);
            SimpleVertex va = g.getVertex("A");
            SimpleVertex vd = g.getVertex("D");


            Map<SimpleVertex, PathEnd<SimpleEdge>> paths = Pathfinding.pathInfo(va, null);
            assertEquals(4, paths.size());
            assertEquals(3, paths.get(vd).distance());
        }


        // Example graph from Prof. Myers's notes
        public static final String graph2 = """
               A -> B 9
               A -> C 14
               A -> D 15
               B -> E 23
               C -> E 17
               C -> D 5
               C -> F 30
               D -> F 20
               D -> G 37
               E -> F 3
               E -> G 20
               F -> G 16""";


        /**
         * Ensures `pathEdges` is a well-formed path: the `dst` of each edge equals the `src` of the
         * subsequent edge, and that the ordered list of all vertices in the path equals
         * `expectedVertices`. Requires `path` is non-empty.
         */
        private void assertPathVertices(List<String> expectedVertices, List<SimpleEdge> pathEdges) {
            ArrayList<String> pathVertices = new ArrayList<>();
            pathVertices.add(pathEdges.getFirst().src().label);
            for (SimpleEdge e : pathEdges) {
                assertEquals(pathVertices.getLast(), e.src().label);
                pathVertices.add(e.dst().label);
            }
            assertIterableEquals(expectedVertices, pathVertices);
        }


        @DisplayName(
                "WHEN a weighted, directed graph is given, THEN `shortestNonBacktracking` returns"
                + "the list of edges in the shortest non-backtracking path from a `src` vertex to a "
                + "`dst` vertex, or null if no such path exists.")
        @Nested
        class testShortestNonBacktrackingPath {


            @DisplayName("When the shortest non-backtracking path consists of multiple edges.")
            @Test
            void testLongPath() {
                SimpleGraph g = SimpleGraph.fromText(graph2);
                List<SimpleEdge> path = Pathfinding.shortestNonBacktrackingPath(g.getVertex("A"),
                        g.getVertex("G"), null);
                assertNotNull(path);
                assertPathVertices(Arrays.asList("A", "C", "E", "F", "G"), path);
            }


            @DisplayName("When the shortest non-backtracking path consists of a single edge.")
            @Test
            void testOneEdgePath() {
                SimpleGraph g = SimpleGraph.fromText("A -> B 4");
                SimpleVertex va = g.getVertex("A");
                SimpleVertex vb = g.getVertex("B");


                List<SimpleEdge> path = Pathfinding.shortestNonBacktrackingPath(va, vb, null);
                assertNotNull(path);
                assertEquals(1, path.size());
                assertEquals(g.getEdge(va, vb), path.get(0));
            }




            @DisplayName("Path is empty when `src` and `dst` are the same.")
            @Test
            void testEmptyPath() {
                SimpleGraph g = SimpleGraph.fromText("A -> A 0");
                SimpleVertex va = g.getVertex("A");


                List<SimpleEdge> path = Pathfinding.shortestNonBacktrackingPath(va, va, null);
                assertNotNull(path);
                assertTrue(path.isEmpty());
            }




            @DisplayName("Path is null when there is not a path from `src` to `dst` (even without "
                         + "accounting for back-tracking.")
            @Test
            void testNoPath() {
                SimpleGraph g = SimpleGraph.fromText("B -> A 2");
                List<SimpleEdge> path = Pathfinding.shortestNonBacktrackingPath(g.getVertex("A"),
                        g.getVertex("B"), null);
                assertNull(path);
            }


            @DisplayName("Path is null when the non-backtracking condition prevents finding a path "
                         + "from `src` to `dst`.")
            @Test
            void testNonBacktrackingPreventsPath() {
                SimpleGraph g = SimpleGraph.fromText("""
                       A -> B 1
                       B -> A 1
                       """);
                SimpleVertex va = g.getVertex("A");
                SimpleVertex vb = g.getVertex("B");


                SimpleEdge prev = g.getEdge(vb, va);


                List<SimpleEdge> path = Pathfinding.shortestNonBacktrackingPath(va, vb, prev);
                assertNull(path);
            }


            @DisplayName(
                    "When the graph includes multiple shortest paths from `src` to `dst`, one of "
                    + "them is returned")
            @Test
            void testMultipleShortestPaths() {
                SimpleGraph g = SimpleGraph.fromText("""
                       A -> B 2
                       A -> C 2
                       B -> D 2
                       C -> D 2
                       """);
                SimpleVertex va = g.getVertex("A");
                SimpleVertex vd = g.getVertex("D");


                List<SimpleEdge> path = Pathfinding.shortestNonBacktrackingPath(va, vd, null);
                assertNotNull(path);
                // Valid paths: A -> B -> D or A -> C -> D
                List<String> vertexLabels = new ArrayList<>();
                vertexLabels.add(path.get(0).src().label);
                for (SimpleEdge e : path) {
                    vertexLabels.add(e.dst().label);
                }
                assertTrue(
                        vertexLabels.equals(Arrays.asList("A", "B", "D")) ||
                        vertexLabels.equals(Arrays.asList("A", "C", "D"))
                );
            }
        }
    }
}



