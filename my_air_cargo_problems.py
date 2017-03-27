from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph
import itertools



class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """
        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = 'T' * len(initial.pos) + 'F' * len(initial.neg)
        
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()
    
    
    def get_actions(self):
        '''
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.
        
        Returns:
        ----------
        list<Action>
            list of Action objects
        '''
        
        #  I'm not sure if this is actually more compact than just using prior
        #  methods, but I couldn't resist the urge to refactor...
        
        #  Slightly brittle, but the format should be clear enough to anyone
        #  working from examples:
        
        FORMULA = 0, NEED_TRUE = 1, NEED_FALSE = 2, MAKE_TRUE = 3, MAKE_FALSE = 4
        
        def enumerate_actions(definition_table: list, vars_table: dict):
            num_clauses = len(definition_table) / 2
            num_vars    = len(vars_table)
            vars_keys   = vars_table.keys()
            actions     = []
            
            def sub_values(clause, combo):
                for i in range(num_vars):
                    clause = clause.replace(vars_keys[i], str(combo[i]))
                return expr(clause)
            
            for combo in itertools.product(**vars_table.values()):
                action_formula = None, needs_true = [], needs_false = [], makes_true = [], makes_false = []
                for i in range(num_clauses):
                    exp    = sub_values(definition_table[i * 2], combo)
                    value  = definition_table[(i * 2) + 1]
                    if   (value == FORMULA   ): action_formula = exp
                    elif (value == NEED_TRUE ): needs_true .append(exp)
                    elif (value == NEED_FALSE): needs_false.append(exp)
                    elif (value == MAKE_TRUE ): makes_true .append(exp)
                    elif (value == MAKE_FALSE): makes_false.append(exp)
                
                action = Action(action_formula, [needs_true, needs_false], [makes_true, makes_false])
                actions.append(action)
            
            return actions
        
        def load_actions(): return enumerate_actions(
                [
                  "Load(cargo, plane, at)", FORMULA,
                  "At(cargo, at)"         , NEED_TRUE,
                  "At(plane, at)"         , NEED_TRUE,
                  "In(cargo, plane)"      , MAKE_TRUE,
                  "At(cargo, at)"         , MAKE_FALSE
                ],
                { "cargo" : self.cargos, "plane" : self.planes, "at" : self.airports }
            )
        
        def unload_actions(): return enumerate_actions(
                [
                  "Unload(cargo, plane, at)", FORMULA,
                  "In(cargo, plane)"        , NEED_TRUE,
                  "At(plane, at)"           , NEED_TRUE,
                  "At(cargo, at)"           , MAKE_TRUE,
                  "In(cargo, plane)"        , MAKE_FALSE
                ],
                { "cargo" : self.cargos, "plane" : self.planes, "at" : self.airports }
            )
        
        def fly_actions(): return enumerate_actions(
                [
                  "Fly(plane, from, to)", FORMULA,
                  "At(plane, from)"     , NEED_TRUE,
                  "At(plane, to)"       , MAKE_TRUE,
                  "At(plane, from)"     , MAKE_FALSE
                ],
                { "plane" : self.planes, "from" : self.airports, "to" : self.airports }
            )
        
        return load_actions() + unload_actions() + fly_actions()
    
    
    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        possible_actions = []
        for action in actions_list:
            for clause in action.precond_pos:
                index = self.state_map.index(clause)
                if state[index] != 'T': continue
            for clause in action.precond_neg:
                index = self.state_map.index(clause)
                if state[index] != 'F': continue
            possible_actions.append(action)
        return possible_actions
    
    
    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given action in
        the given state. The action must be one of self.actions(state).
        
        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        new_state = list(self.state_map)
        for clause in action.effect_add:
            index = self.state_map.index(clause)
            new_state[index] = 'T'
        for clause in action.effect_rem:
            index = self.state_map.index(clause)
            new_state[index] = 'F'
        return str(new_state)
    
    
    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        for clause in self.goal:
            index = self.state_map.index(clause)
            if state[index] != 'T': return False
        return True
    
    
    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const
    
    
    def h_pg_levelsum(self, node: Node):
        '''
        This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        '''
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum
    
    
    def h_ignore_preconditions(self, node: Node):
        '''
        This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        '''
        # TODO implement (see Russell-Norvig Ed-3 10.2.3  or Russell-Norvig Ed-2 11.2)
        count = 0
        return count



def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p2() -> AirCargoProblem:
    # TODO implement Problem 2 definition
    pass


def air_cargo_p3() -> AirCargoProblem:
    # TODO implement Problem 3 definition
    pass
