package model;

import java.awt.Color;
import java.util.Random;
import model.MazeGraph.IPair;
import model.MazeGraph.MazeVertex;

public class Clyde extends Ghost{

    /**
     * Clyde's random number generator
     */
    Random random;

    /**
     * Construct a ghost associated to the given `model` with specified color and initial delay
     *
     * @param model
     */
    public Clyde(GameModel model, Random random) {
        super(model, Color.ORANGE, 8000);
        this.random = random;
    }

    /**
     * Return the vertex that this ghost is targeting
     */
    @Override
    protected MazeVertex target() {
        if (state == GhostState.CHASE){
            MazeVertex pacManLocation = model.pacMann().nearestVertex();
            MazeVertex clydeLocation = this.nearestVertex();
            IPair pacManCoords = pacManLocation.loc();
            IPair clydeCoords = clydeLocation.loc();

            double distance = Math.sqrt(Math.pow(pacManCoords.i() - clydeCoords.i(),2) + Math.pow(pacManCoords.j() - clydeCoords.j(),2));

            if (distance>10){return model.pacMann().nearestVertex();}
            else {
                int i = random.nextInt(model.width());
                int j = random.nextInt(model.height());
                return model.graph().closestTo(i, j);

            }


        } else if (state == GhostState.FLEE) {
            return model.graph().closestTo(model.width()-3,model.height()-3);
        }
        return null;
    }
}
