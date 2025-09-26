package model;

import model.MazeGraph.MazeEdge;
import model.MazeGraph.MazeVertex;

//TODO priority 1: document this
public class PacMannManual extends PacMann {
    /**
     * Keeps track of the game model
     */
    GameModel gameModel;

    public PacMannManual(GameModel gameModel){
        super(gameModel);
        this.gameModel = gameModel;
    }

    public MazeEdge nextEdge(){
        MazeVertex v = gameModel.pacMann().nearestVertex();

        MazeEdge e = v.edgeInDirection(gameModel.playerCommand());
        if (e != null){
            return e;
        }
        e = v.edgeInDirection(gameModel.pacMann().currentEdge().direction());
        return e;
    }

}
