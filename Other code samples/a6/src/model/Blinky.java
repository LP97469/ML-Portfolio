package model;

import java.awt.Color;
import model.MazeGraph.MazeVertex;

public class Blinky extends Ghost{

    public Blinky(GameModel gameModel) {
        super(gameModel, Color.RED, 2000);
    }

    /**
     * Return the vertex that this ghost is targeting
     */
    @Override
    protected MazeVertex target() {
        if (state == GhostState.CHASE){
            return model.pacMann().nearestVertex();
        } else if (state == GhostState.FLEE) {
            return model.graph().closestTo(2,2);
        }
        return null;
    }
}
