package model;

import graph.Vertex;
import java.awt.Color;
import model.MazeGraph.Direction;
import model.MazeGraph.IPair;
import model.MazeGraph.MazeEdge;
import model.MazeGraph.MazeVertex;

public class Pinky extends Ghost{

    /**
     * Construct a ghost associated to the given `model` with specified color and initial delay
     *
     * @param model
     */
    public Pinky(GameModel model) {
        super(model, Color.PINK, 4000);
    }

    /**
     * Return the vertex that this ghost is targeting
     */
    @Override
    protected MazeVertex target() {
        if (state == GhostState.CHASE){
            MazeVertex pacManLocation = model.pacMann().nearestVertex();
            Direction pacManDirection = model.pacMann().currentEdge().direction();
            IPair coords = pacManLocation.loc();
            switch (pacManDirection){
                case UP:
                    return model.graph().closestTo(coords.i(), coords.j()-3);
                case DOWN:
                    return model.graph().closestTo(coords.i(), coords.j()+3);
                case LEFT:
                    return model.graph().closestTo(coords.i()-3, coords.j());
                case RIGHT:
                    return model.graph().closestTo(coords.i()+3, coords.j());

            }


            return null;
        } else if (state == GhostState.FLEE) {
            return model.graph().closestTo(model.width()-3,2);
        }
        return null;
    }
}
