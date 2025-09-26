package model;

import java.awt.Color;
import model.MazeGraph.IPair;
import model.MazeGraph.MazeVertex;

public class Inky extends Ghost{

    /**
     * Construct a ghost associated to the given `model` with specified color and initial delay
     *
     * @param model
     */
    public Inky(GameModel model) {
        super(model, Color.CYAN, 6000);
    }

    /**
     * Return the vertex that this ghost is targeting
     */
    @Override
    protected MazeVertex target() {
        if (state == GhostState.CHASE){
            MazeVertex pacManLocation = model.pacMann().nearestVertex();
            MazeVertex blinkyLocation = model.blinky().nearestVertex();
            IPair pacManCoords = pacManLocation.loc();
            IPair blinkyCoords = blinkyLocation.loc();

            return model.graph().closestTo((pacManCoords.i()+ blinkyCoords.i())/2,(pacManCoords.j()+ blinkyCoords.j())/2);

        } else if (state == GhostState.FLEE) {
            return model.graph().closestTo(2,model.height()-3);
        }
        return null;
    }
}
