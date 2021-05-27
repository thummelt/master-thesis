from src.modules.state import State
from src.modules.decision import Decision
from src.modules.transition import Transition
from src.modules.analysis import Analysis

from typing import List, Dict


## Represents VI algorithm

class SolutionAlgorithms:

    # Variables
    conv : dict
    epsilon : float


    def __init__(self):
        pass

    def performStandardVI(self, states: List[State], decisions: Dict[State, List[Decision]]) -> bool:
        return True
        #
        #int iterationCounter = 0;
#
#
        #Map<State, Boolean> convergence = new HashMap<State, Boolean>();
        #states.forEach(s -> {
        #    convergence.put(s, false);
        #});
#
        #while(!convergence.values().stream().allMatch(val -> val == true)){
        #    iterationCounter ++;
        #    states.stream().forEach(s -> {
        #        double oldV = s.getValue(-1);
        #        
        #        double newV = decisions.get(s).stream().map(d -> {
        #            // Calculate contribution
        #            double contribution = d.getX_c().size();
#
        #            // Perform transition and obtain value of last iteration from destination states
        #            double expectedDestStateVal = transitions.get(s).get(d).stream().map(t -> t.getProbability()*t.getDestination().getValue(-1)).reduce(Double::sum).orElse(0.0);
#
        #            return contribution + expectedDestStateVal;
        #        }).max(Double::compare).get();
#
        #        // Store new value and update convergence map
        #        s.setCurrentValue(newV);     
        #        convergence.put(s, s.hasConverged());
        #    });
#
        #    // Update state values for last iteration
        #    states.parallelStream().forEach(s -> s.nextIterationPrep());
        #}       
        
        #logger.log(Level.INFO, String.format("Finished in %d iterations", iterationCounter));
        #return iterationCounter;
    #}

