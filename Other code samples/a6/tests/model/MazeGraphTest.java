package model;

import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import model.MazeGraph.Direction;
import model.MazeGraph.MazeEdge;
import model.MazeGraph.IPair;
import model.MazeGraph.MazeVertex;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import util.GameMap;
import util.MazeGenerator.TileType;

public class MazeGraphTest {

    /* Note, to conform to the precondition of the `MazeGraph` constructor, make sure that any
     * TileType arrays that you construct contain a `PATH` tile at index [2][2] and represent a
     * single, orthogonally connected component of `PATH` tiles. */

    /**
     * Create a game map with tile types corresponding to the letters on each line of `template`.
     * 'w' = WALL, 'p' = PATH, and 'g' = GHOSTBOX.  The letters of `template` must form a rectangle.
     * Elevations will be a gradient from the top-left to the bottom-right corner with a horizontal
     * slope of 2 and a vertical slope of 1.
     */
    static GameMap createMap(String template) {
        Scanner lines = new Scanner(template);
        ArrayList<ArrayList<TileType>> lineLists = new ArrayList<>();

        while (lines.hasNextLine()) {
            ArrayList<TileType> lineList = new ArrayList<>();
            for (char c : lines.nextLine().toCharArray()) {
                switch (c) {
                    case 'w' -> lineList.add(TileType.WALL);
                    case 'p' -> lineList.add(TileType.PATH);
                    case 'g' -> lineList.add(TileType.GHOSTBOX);
                }
            }
            lineLists.add(lineList);
        }

        int height = lineLists.size();
        int width = lineLists.getFirst().size();

        TileType[][] types = new TileType[width][height];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                types[i][j] = lineLists.get(j).get(i);
            }
        }

        double[][] elevations = new double[width][height];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                elevations[i][j] = (2.0 * i + j);
            }
        }
        return new GameMap(types, elevations);
     }

    @DisplayName("WHEN a GameMap with exactly one path tile in position [2][2] is passed into the "
            + "MazeGraph constructor, THEN a graph with one vertex is created.")
    @Test
    void testOnePathCell() {
        GameMap map = createMap("""
                wwwww
                wwwww
                wwpww
                wwwww
                wwwww""");
        MazeGraph graph = new MazeGraph(map);
        Map<IPair, MazeVertex> vertices = new HashMap<>();
        graph.vertices().forEach(v -> vertices.put(v.loc(), v));

        assertEquals(1, vertices.size());
        assertTrue(vertices.containsKey(new IPair(2, 2)));
    }

    @DisplayName("WHEN a GameMap with exactly two horizontally adjacent path tiles is passed into "
            + "the MazeGraph constructor, THEN a graph with two vertices is created in which the two "
            + "vertices are connected by two directed edges with weights determined by evaluating "
            + "`MazeGraph.edgeWeight` on their elevations.")
    @Test
    void testTwoPathCellsHorizontal() {
        GameMap map = createMap("""
                wwwww
                wwwww
                wwppw
                wwwww
                wwwww""");
        MazeGraph graph = new MazeGraph(map);
        Map<IPair, MazeVertex> vertices = new HashMap<>();
        graph.vertices().forEach(v -> vertices.put(v.loc(), v));

        // graph contains two vertices with the correct locations
        assertEquals(2, vertices.size());
        IPair left = new IPair(2, 2);
        IPair right = new IPair(3, 2);
        assertTrue(vertices.containsKey(left));
        assertTrue(vertices.containsKey(right));

        MazeVertex vl = vertices.get(left);
        MazeVertex vr = vertices.get(right);

        // left vertex has one edge to the vertex to its right
        assertNull(vl.edgeInDirection(Direction.LEFT));
        assertNull(vl.edgeInDirection(Direction.UP));
        assertNull(vl.edgeInDirection(Direction.DOWN));
        MazeEdge l2r = vl.edgeInDirection(Direction.RIGHT);
        assertNotNull(l2r);

        // edge from left to right has the correct fields
        double lElev = map.elevations()[2][2];
        double rElev = map.elevations()[3][2];
        assertEquals(vl, l2r.src());
        assertEquals(vr, l2r.dst());
        assertEquals(Direction.RIGHT, l2r.direction());
        assertEquals(MazeGraph.edgeWeight(lElev, rElev), l2r.weight());

        // right vertex has one edge to the vertex to its left with the correct fields
        assertNull(vr.edgeInDirection(Direction.RIGHT));
        assertNull(vr.edgeInDirection(Direction.UP));
        assertNull(vr.edgeInDirection(Direction.DOWN));
        MazeEdge r2l = vr.edgeInDirection(Direction.LEFT);
        assertNotNull(r2l);
        assertEquals(vr, r2l.src());
        assertEquals(vl, r2l.dst());
        assertEquals(Direction.LEFT, r2l.direction());
        assertEquals(MazeGraph.edgeWeight(rElev, lElev), r2l.weight());
    }

    @DisplayName("WHEN a GameMap with exactly two vertically adjacent path tiles is passed into "
            + "the MazeGraph constructor, THEN a graph with two vertices is created in which the two "
            + "vertices are connected by two directed edges with weights determined by evaluating "
            + "`MazeGraph.edgeWeight` on their elevations.")
    @Test
    void testTwoPathCellsVertical() {
        // TODO 2a done: Complete this test case
        TileType[][] tiles = new TileType[1][2];
        tiles[0][0] = TileType.PATH;
        tiles[0][1] = TileType.PATH;
        double[][] elevations = new double[1][2];
        elevations[0][0] = 0;
        elevations[0][1] = 0;

        GameMap newMap = new GameMap(tiles, elevations);

        MazeGraph mazeGraph = new MazeGraph(newMap);
        //System.out.println(mazeGraph.vertices());
        /// string below manually verified
        assertEquals("[MazeVertex{loc=IPair[i=0, j=0], edges=[DOWN=1.0, UP=1.0]}, MazeVertex{loc=IPair[i=0, j=1], edges=[DOWN=1.0, UP=1.0]}]", mazeGraph.vertices().toString());
        //fail("Testcase has not been implemented");
    }

    @DisplayName("WHEN a GameMap includes two path tiles in the first and last column of the same "
            + "row, THEN (tunnel) edges are created between these tiles with the correct properties.")
    @Test
    void testHorizontalTunnelEdgeCreation() {
        // TODO 2b done: Complete this test case
        TileType[][] tiles = new TileType[2][1];
        tiles[0][0] = TileType.PATH;
        tiles[1][0] = TileType.PATH;
        double[][] elevations = new double[2][1];
        elevations[0][0] = 0;
        elevations[1][0] = 0;

        GameMap newMap = new GameMap(tiles, elevations);

        MazeGraph mazeGraph = new MazeGraph(newMap);
        System.out.println(mazeGraph.vertices());
        /// string below manually verified
        assertEquals("[MazeVertex{loc=IPair[i=0, j=0], edges=[LEFT=1.0, RIGHT=1.0]}, MazeVertex{loc=IPair[i=1, j=0], edges=[LEFT=1.0, RIGHT=1.0]}]", mazeGraph.vertices().toString());
    }

    @DisplayName("WHEN a GameMap includes a cyclic connected component of path tiles with a "
            + "non-path tiles in the middle, THEN its graph includes edges between all adjacent "
            + "pairs of vertices.")
    @Test
    void testCyclicPaths() {
        GameMap map = createMap("""
                wwwwwww
                wwwwwww
                wwpppww
                wwpwpww
                wwpppww
                wwwwwww""");
        MazeGraph graph = new MazeGraph(map);

        // TODO 2c done: Complete this test case
        //System.out.println(graph.vertices());
        /// string below manually verified
        assertEquals("[MazeVertex{loc=IPair[i=2, j=2], edges=[DOWN=1.75, RIGHT=1.75]}, MazeVertex{loc=IPair[i=4, j=4], edges=[LEFT=0.25, UP=0.25]}, MazeVertex{loc=IPair[i=2, j=3], edges=[DOWN=1.75, UP=0.25]}, MazeVertex{loc=IPair[i=3, j=4], edges=[LEFT=0.25, RIGHT=1.75]}, MazeVertex{loc=IPair[i=2, j=4], edges=[RIGHT=1.75, UP=0.25]}, MazeVertex{loc=IPair[i=4, j=2], edges=[LEFT=0.25, DOWN=1.75]}, MazeVertex{loc=IPair[i=3, j=2], edges=[LEFT=0.25, RIGHT=1.75]}, MazeVertex{loc=IPair[i=4, j=3], edges=[DOWN=1.75, UP=0.25]}]", graph.vertices().toString());
        //fail("Testcase has not been implemented");
    }

    // TODO 2d done: Add at least two additional test cases that test other distinct path structures.
    //  It is crucial that your graph is being linked together correctly, otherwise the later
    //  portions of the assignment will break with strange behaviors.

    @DisplayName("WHEN a GameMap includes no connected paths THEN its graph should have no edges")
    @Test
    void testNoPaths() {
        GameMap map = createMap("""
                pwwwwww
                wpwwwww
                wwpwwww
                wwwpwww
                wwwwpww
                wwwwwpw""");
        MazeGraph graph = new MazeGraph(map);

        //System.out.println(graph.vertices());
        /// string below manually verified
        assertEquals("[MazeVertex{loc=IPair[i=0, j=0], edges=[]}, MazeVertex{loc=IPair[i=1, j=1], edges=[]}, MazeVertex{loc=IPair[i=2, j=2], edges=[]}, MazeVertex{loc=IPair[i=3, j=3], edges=[]}, MazeVertex{loc=IPair[i=4, j=4], edges=[]}, MazeVertex{loc=IPair[i=5, j=5], edges=[]}]", graph.vertices().toString());
        //fail("Testcase has not been implemented");
    }

    @DisplayName("WHEN a GameMap includes paths along the edges of the map THEN the board should loop correctly")
    @Test
    void testEdgePaths() {
        GameMap map = createMap("""
                ppppppp
                pwwwwwp
                pwwwwwp
                pwwwwwp
                pwwwwwp
                ppppppp""");
        MazeGraph graph = new MazeGraph(map);

        //System.out.println(graph.vertices());
        /// string below manually verified
        assertEquals("[MazeVertex{loc=IPair[i=0, j=0], edges=[UP=1.75, RIGHT=1.75, LEFT=1.75, DOWN=1.75]}, MazeVertex{loc=IPair[i=5, j=5], edges=[RIGHT=1.75, LEFT=0.25, DOWN=0.25]}, MazeVertex{loc=IPair[i=0, j=1], edges=[UP=0.25, LEFT=1.75, DOWN=1.75]}, MazeVertex{loc=IPair[i=4, j=5], edges=[RIGHT=1.75, LEFT=0.25, DOWN=0.25]}, MazeVertex{loc=IPair[i=0, j=2], edges=[UP=0.25, LEFT=1.75, DOWN=1.75]}, MazeVertex{loc=IPair[i=3, j=5], edges=[RIGHT=1.75, LEFT=0.25, DOWN=0.25]}, MazeVertex{loc=IPair[i=0, j=3], edges=[UP=0.25, LEFT=1.75, DOWN=1.75]}, MazeVertex{loc=IPair[i=2, j=5], edges=[RIGHT=1.75, LEFT=0.25, DOWN=0.25]}, MazeVertex{loc=IPair[i=0, j=4], edges=[UP=0.25, LEFT=1.75, DOWN=1.75]}, MazeVertex{loc=IPair[i=1, j=5], edges=[RIGHT=1.75, LEFT=0.25, DOWN=0.25]}, MazeVertex{loc=IPair[i=0, j=5], edges=[UP=0.25, RIGHT=1.75, LEFT=1.75, DOWN=0.25]}, MazeVertex{loc=IPair[i=6, j=0], edges=[UP=1.75, RIGHT=0.25, LEFT=0.25, DOWN=1.75]}, MazeVertex{loc=IPair[i=5, j=0], edges=[UP=1.75, RIGHT=1.75, LEFT=0.25]}, MazeVertex{loc=IPair[i=6, j=1], edges=[UP=0.25, RIGHT=0.25, DOWN=1.75]}, MazeVertex{loc=IPair[i=4, j=0], edges=[UP=1.75, RIGHT=1.75, LEFT=0.25]}, MazeVertex{loc=IPair[i=6, j=2], edges=[UP=0.25, RIGHT=0.25, DOWN=1.75]}, MazeVertex{loc=IPair[i=3, j=0], edges=[UP=1.75, RIGHT=1.75, LEFT=0.25]}, MazeVertex{loc=IPair[i=6, j=3], edges=[UP=0.25, RIGHT=0.25, DOWN=1.75]}, MazeVertex{loc=IPair[i=2, j=0], edges=[UP=1.75, RIGHT=1.75, LEFT=0.25]}, MazeVertex{loc=IPair[i=6, j=4], edges=[UP=0.25, RIGHT=0.25, DOWN=1.75]}, MazeVertex{loc=IPair[i=1, j=0], edges=[UP=1.75, RIGHT=1.75, LEFT=0.25]}, MazeVertex{loc=IPair[i=6, j=5], edges=[UP=0.25, RIGHT=0.25, LEFT=0.25, DOWN=0.25]}]", graph.vertices().toString());
        //fail("Testcase has not been implemented");
    }

}
