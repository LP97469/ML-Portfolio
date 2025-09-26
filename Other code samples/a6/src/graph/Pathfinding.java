package graph;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class Pathfinding {

    /**
     * Represents a path ending at `lastEdge.end()` along with its total weight (distance).
     */
    record PathEnd<E extends Edge<?>>(double distance, E lastEdge) {

    }

    /**
     * Returns a list of `E` edges comprising the shortest non-backtracking path from vertex `src`
     * to vertex `dst`. A non-backtracking path never contains two consecutive edges between the
     * same two vertices (e.g., v -> w -> v). As a part of this requirement, the first edge in the
     * returned path cannot back-track `previousEdge` (when `previousEdge` is not null). If there is
     * not a non-backtracking path from `src` to `dst`, then null is returned. Requires that if `E
     * != null` then `previousEdge.dst().equals(src)`.
     */
    public static <V extends Vertex<E>, E extends Edge<V>> List<E> shortestNonBacktrackingPath(
            V src, V dst, E previousEdge) {

        Map<V, PathEnd<E>> paths = pathInfo(src, previousEdge);
        return paths.containsKey(dst) ? pathTo(paths, src, dst) : null;
    }

    /**
     * Returns a map that associates each vertex reachable from `src` along a non-backtracking path
     * with a `PathEnd` object. The `PathEnd` object summarizes relevant information about the
     * shortest non-backtracking path from `src` to that vertex. A non-backtracking path never
     * contains two consecutive edges between the same two vertices (e.g., v -> w -> v). As a part
     * of this requirement, the first edge in the returned path cannot backtrack `previousEdge`
     * (when `previousEdge` is not null). Requires that if `E != null` then
     * `previousEdge.dst().equals(src)`.
     */
    static <V extends Vertex<E>, E extends Edge<V>> Map<V, PathEnd<E>> pathInfo(V src,
            E previousEdge) {

        assert previousEdge == null || previousEdge.dst().equals(src);

        // Associate vertex labels with info about the shortest-known path from `start` to that
        // vertex.  Populated as vertices are discovered (not as they are settled).
        Map<V, PathEnd<E>> pathInfo = new HashMap<>();

        // TODO 11a done but unchecked: Complete the implementation of this method according to its specification. Your
        //  implementation should make use of Dijkstra's algorithm (modified to prevent path back-
        //  tracking), creating a `MinPQueue` to manage the "frontier" set of vertices, and settling
        //  the vertices in this frontier in increasing distance order.
        MinPQueue<Vertex<E>> frontier = new MinPQueue<>();
        Map<Vertex<E>, Double> knownDistances = new HashMap<>();
        Map<V,E> lastEdge = new HashMap<>();


        frontier.addOrUpdate(src, 0.0);
        knownDistances.put(src, 0.0);
        lastEdge.put(src, previousEdge);


        while (!frontier.isEmpty()) {
            Vertex<E> v = frontier.remove();
            double d = knownDistances.get(v);
            E incoming = lastEdge.get(v);

            pathInfo.put((V) v, new PathEnd<>(d, incoming));

            for (E edge : v.outgoingEdges()) {
                V w = edge.dst();
                if (incoming != null
                    && incoming.dst().equals(v)
                    && incoming.src().equals(w)) {
                    continue;
                }
                double nd = d + edge.weight();
                if (!knownDistances.containsKey(w) || nd < knownDistances.get(w)) {
                    knownDistances.put(w, nd);
                    lastEdge.put(w, edge);
                    frontier.addOrUpdate(w, nd);
                }
            }
        }


        return pathInfo;
    }

    /**
     * Return the list of edges in the shortest non-backtracking path from `src` to `dst`, as
     * summarized by the given `pathInfo` map. Requires `pathInfo` conforms to the specification as
     * documented by the `pathInfo` method; it must contain backpointers for the shortest
     * non-backtracking paths from `src` to all reachable vertices.
     */
    static <V, E extends Edge<V>> List<E> pathTo(Map<V, PathEnd<E>> pathInfo, V src, V dst) {
        // Prefer a linked list for efficient prepend (alternatively, could append, then reverse
        // before returning)
        List<E> path = new LinkedList<>();

        // TODO 11b priority 0: Complete this implementation of this method according to its specification.
        //throw new UnsupportedOperationException();
        V current = dst;
        while(true) {
            if (current == src) {
                break;
            }
            PathEnd<E> end = pathInfo.get(current);
            E edge = end.lastEdge;
            path.addFirst(edge);
            V prev = edge.src();
            current = prev;

        }


        return path;
    }
}
