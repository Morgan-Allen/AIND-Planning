#from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import Node, Problem
from aimacode.utils import expr
from lp_utils import FluentState
from my_planning_graph import PlanningGraph
from my_planning_graph import ReverseNeedsLevelLookup
import itertools
import random


'''
Keys used to help define actions and their effects or conditions.
'''
FORMULA    = 0
NEED_TRUE  = 1
NEED_FALSE = 2
MAKE_TRUE  = 3
MAKE_FALSE = 4

def enumerate_actions(definition_table: list, vars_table: dict):
    '''
    Enumerates over all possible combinations of action, based on a list of
    arguments supplied to define the parameters, conditions and effects of
    the schema, with substitution by a range of values for each variable.
    
    Example:
    enumerate_actions(
      [
        "Eat(person, food, venue)", FORMULA,
        "Has(venue, food)"        , NEED_TRUE,
        "At(venue, person)"       , NEED_TRUE,
        "Has(venue, food)"        , MAKE_FALSE
        "Full(person)"            , MAKE_TRUE
      ]
      {
        "person" : ["Heinz", "Luigi", "Bill"],
        "food"   : ["Bratwurst", "Pasta", "Hot Dogs"],
        "venue"  : ["Home", "Restaurant"]
    )
    ...producing "Eat(Heinz, Pasta, Home)", "Eat(Bill, Bratwurst, Restaurant)",
    etc., for all 18 possible substitutions.
    
    :param definition_table : an alternating list of schema (defined in string
                              format), and a key to specify it's function in
                              defining the action.  Must be either:
                              
                              FORMULA    -specifies name and parameters
                              NEED_TRUE  -a positive precondition
                              NEED_FALSE -a negative precondition
                              MAKE_TRUE  -an additive/positive effect
                              MAKE_FALSE -a subtractive/negative effect
                              
    :param vars_table       : a dictionary mapping variables (as strings) to
                              a list of possible values.  All non-repeating
                              combinations of these values are then
                              substituted for these variables in the schema
                              above.
    
    :return                 : The resultant list of concrete actions derived
                              from the specified schema, variables and values.
    '''
    num_clauses = int(len(definition_table) / 2)
    num_vars    = len(vars_table)
    vars_keys   = list(vars_table.keys())
    actions     = []
    
    def sub_values(clause, combo):
        for i in range(num_vars):
            clause = clause.replace(vars_keys[i], str(combo[i]))
        return expr(clause)
    
    #  NOTE:  We skip over combinations that feature the same values more than
    #  once- e.g, flying to and from the same airport...
    for combo in itertools.product(*vars_table.values()):
        if len(combo) != len(set(combo)): continue
        action_formula = None
        needs_true  = []
        needs_false = []
        makes_true  = []
        makes_false = []
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



class AirCargoProblem(Problem):
    
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        '''
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
        '''
        self.state_map        = initials = initial.pos + initial.neg
        self.indices          = dict((initials[i], i) for i in range(len(initials)))
        self.initial_state_TF = 'T' * len(initial.pos) + 'F' * len(initial.neg)
        
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        
        self.cargos   = cargos
        self.planes   = planes
        self.airports = airports
        self.actions_list = self.get_actions()
        
        self.heuristic_lookup = None
    
    
    def print_action(self, action):
        '''
        Prints the given action in terms of it's name, arguments, conditions
        and effects.
        '''
        print(
            " ", action.name, action.args,
            "\n    Needs True:  ", action.precond_pos,
            "\n    Needs False: ", action.precond_neg,
            "\n    Makes True:  ", action.effect_add,
            "\n    Makes False: ", action.effect_rem
        )
    
    
    def print_state(self, state):
        '''
        Prints the given state as a mapping of concrete facts to their truth
        values:
        '''
        print("\nPrinting state: ")
        for i in range(len(self.state_map)):
            print("  ", self.state_map[i], ": ", state[i])
    
    
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
    
    
    def matching_clauses(self, clauses, value, state):
        return [m for m in clauses if state[self.indices[m]] == value]
    
    
    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.
        
        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        possible_actions = []
        for action in self.actions_list:
            if self.matching_clauses(action.precond_pos, 'F', state): continue
            if self.matching_clauses(action.precond_neg, 'T', state): continue
            possible_actions.append(action)
        return possible_actions
    
    
    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given action in
        the given state. The action must be one of self.actions(state).
        
        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        new_state = list(state)
        for clause in action.effect_add: new_state[self.indices[clause]] = 'T'
        for clause in action.effect_rem: new_state[self.indices[clause]] = 'F'
        return "".join(new_state)
    
    
    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached
        
        :param state: str representing state
        :return: bool
        """
        if self.matching_clauses(self.goal, 'F', state): return False
        return True
    
    
    def h_1(self, node: Node):
        """
        Simple constant-value heuristic for baseline comparison.
        """
        h_const = 1
        return h_const
    
    
    def h_unmet_goals(self, node: Node):
        '''
        This heuristic performs a cruder estimate of 'work remaining' than the
        ignore-preconditions heuristic, simply by counting the number of unmet
        goals.
        '''
        return len(self.matching_clauses(self.goal, 'F', node.state))
    
    
    def h_ignore_preconditions(self, node: Node):
        '''
        This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        '''
        count = 0
        unmet_goals = self.matching_clauses(self.goal, 'F', node.state)
        
        for action in self.actions_list:
            for clause in action.effect_add:
                if clause in unmet_goals:
                    count += 1
                    unmet_goals.remove(clause)
                    if len(unmet_goals) == 0: return count
        
        return count
    
    
    def h_reverse_levelsum(self, node: Node):
        '''
        This heuristic uses a loose 'reversal' of the Planning Graph approach
        to estimate the sum of actions required to satisfy the problem's goal-
        state from the given node's state.  Since this approach works backward
        from the problem's goals, which remain constant, the 'graph' can be
        cached for re-use throughout the search.
        '''
        if self.heuristic_lookup == None:
            self.heuristic_lookup = ReverseNeedsLevelLookup(self)
        return self.heuristic_lookup.h_levelsum(node.state)
    
    
    def h_reverse_setlevel(self, node: Node):
        '''
        Similar to the above, but using an adapted version of the set-level
        heuristic described in chapter 11 of AIMA 2E.
        '''
        if self.heuristic_lookup == None:
            self.heuristic_lookup = ReverseNeedsLevelLookup(self)
        return self.heuristic_lookup.h_setlevel(node.state)
    
    
    def h_pg_levelsum(self, node: Node):
        '''
        This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        '''
        pg = PlanningGraph(self, node.state)
        level = pg.h_levelsum()
        if level == False: return float("-inf")
        return level
    
    
    def h_pg_setlevel(self, node: Node):
        '''
        Similar to the above, but using the set-level heuristic described in
        chapter 11 of AIMA 2E.
        '''
        pg = PlanningGraph(self, node.state)
        level = pg.h_setlevel()
        if level == False: return float("-inf")
        return level



def enumerate_air_cargo_problem(cargos, planes, airports, facts, goals):
    '''
    Constructs all the concrete actions required for a problem from some
    relatively simple string-based formulae.
    
    :param cargos  : a list of all possible cargo IDs (strings)
    :param planes  : a list of all possible plane IDs (strings)
    :param airports: a list of all possible airport IDs (strings)
    :param facts   : a list of all facts which are true in the initial state
                     (all others are initialised as false by default), also
                     as strings
    :param goals   : a list of all facts in the goal-state, also as strings
    
    :return        : an AirCargoProblem with a suitable range of 'At' and 'In'
                     expressions specified for it's initial state and goals.
    '''
    
    #  We generate 'dummy actions' with no effects and no conditions, simply
    #  to get the full range of possible facts from their signature schema:
    cargo_at_actions = enumerate_actions(["At(cargo, port)" , FORMULA], { "cargo" : cargos, "port" : airports })
    plane_at_actions = enumerate_actions(["At(plane, port)" , FORMULA], { "plane" : planes, "port" : airports })
    cargo_in_actions = enumerate_actions(["In(cargo, plane)", FORMULA], { "cargo" : cargos, "plane": planes   })
    all_actions      = cargo_at_actions + plane_at_actions + cargo_in_actions
    
    #  All the string arguments are converted to expressions before being used
    #  to initialise and return an AirCargoProblem.
    all_fluents = [expr(action.name+str(action.args)) for action in all_actions]
    facts       = [expr(fact) for fact in facts]
    goals       = [expr(goal) for goal in goals]
    negatives   = [fluent for fluent in all_fluents if not fluent in facts]
    return AirCargoProblem(cargos, planes, airports, FluentState(facts, negatives), goals)


def air_cargo_p1() -> AirCargoProblem:
    return enumerate_air_cargo_problem(
        cargos   = ['C1', 'C2'],
        planes   = ['P1', 'P2'],
        airports = ['JFK', 'SFO'],
        facts = [
            'At(C1, SFO)', 'At(C2, JFK)',
            'At(P1, SFO)', 'At(P2, JFK)'
        ],
        goals = ['At(C1, JFK)', 'At(C2, SFO)']
    )


def air_cargo_p2() -> AirCargoProblem:
    return enumerate_air_cargo_problem(
        cargos   = ['C1', 'C2', 'C3'],
        planes   = ['P1', 'P2', 'P3'],
        airports = ['JFK', 'SFO', 'ATL'],
        facts = [
            'At(C1, SFO)', 'At(C2, JFK)', 'At(C3, ATL)',
            'At(P1, SFO)', 'At(P2, JFK)', 'At(P3, ATL)'
        ],
        goals = ['At(C1, JFK)', 'At(C2, SFO)', 'At(C3, SFO)']
    )


def air_cargo_p3() -> AirCargoProblem:
    return enumerate_air_cargo_problem(
        cargos   = ['C1', 'C2', 'C3', 'C4'],
        planes   = ['P1', 'P2'],
        airports = ['JFK', 'SFO', 'ATL', 'ORD'],
        facts = [
            'At(C1, SFO)', 'At(C2, JFK)', 'At(C3, ATL)', 'At(C4, ORD)',
            'At(P1, SFO)', 'At(P2, JFK)'
        ],
        goals = ['At(C1, JFK)', 'At(C3, JFK)', 'At(C2, SFO)', 'At(C4, SFO)']
    )


def air_cargo_random_problem() -> AirCargoProblem:
    '''
    Constructs a randomised air-cargo problem, generally for single-run tests,
    with varying numbers and locations for airports, planes and cargo.
    '''
    cargos   = ['C1', 'C2', 'C3', 'C4', 'C5']     [0:random.randrange(3, 5)]
    planes   = ['P1', 'P2', 'P3', 'P4']           [0:random.randrange(1, 4)]
    airports = ['JFK', 'SFO', 'ATL', 'ORD', 'MIA'][0:random.randrange(3, 5)]
    
    facts = []
    goals = []
    for plane in planes:
        at = random.choice(airports)
        facts.append('At({}, {})'.format(plane, at))
    for cargo in cargos:
        at = random.choice(airports)
        facts.append('At({}, {})'.format(cargo, at))
        goes = random.choice([a for a in airports if a != at])
        goals.append('At({}, {})'.format(cargo, goes))
    
    print("\nGenerated Random Problem:")
    print("  Facts:", facts)
    print("  Goals:", goals)
    return enumerate_air_cargo_problem(cargos, planes, airports, facts, goals)

