from aimacode.planning import Action
from aimacode.search import Problem
from aimacode.utils import expr
from lp_utils import decode_state



class PgNode():
    '''
    Base class for planning graph nodes.
    
    includes instance sets common to both types of nodes used in a planning graph
    parents: the set of nodes in the previous level
    children: the set of nodes in the subsequent level
    mutex: the set of sibling nodes that are mutually exclusive with this node
    '''
    
    def __init__(self):
        self.parents = set()
        self.children = set()
        self.mutex = set()
    
    
    def is_mutex(self, other) -> bool:
        '''
        Boolean test for mutual exclusion
        
        :param other: PgNode
            the other node to compare with
        :return: bool
            True if this node and the other are marked mutually exclusive (mutex)
        '''
        return other in self.mutex
    
    
    def show(self):
        '''
        Helper print for debugging shows counts of parents, children, siblings
        
        :return:
            print only
        '''
        print("{} parents".format(len(self.parents)))
        print("{} children".format(len(self.children)))
        print("{} mutex".format(len(self.mutex)))


class PgNode_s(PgNode):
    '''
    A planning graph node representing a state (literal fluent) from a planning
    problem.
    
    Args:
    ----------
    symbol : str
        A string representing a literal expression from a planning problem
        domain.
    
    is_pos : bool
        Boolean flag indicating whether the literal expression is positive or
        negative.
    '''
    
    def __init__(self, symbol: expr, is_pos: bool):
        '''
        S-level Planning Graph node constructor
        
        :param symbol: expr
        :param is_pos: bool
        Instance variables calculated:
            literal: expr
                    fluent in its literal form including negative operator if applicable
        Instance variables inherited from PgNode:
            parents: set of nodes connected to this node in previous A level; initially empty
            children: set of nodes connected to this node in next A level; initially empty
            mutex: set of sibling S-nodes that this node has mutual exclusion with; initially empty
        '''
        PgNode.__init__(self)
        self.symbol = symbol
        self.is_pos = is_pos
    
    
    def show(self):
        '''
        Helper print for debugging shows literal plus counts of parents, children, siblings
        
        :return:
            print only
        '''
        print("\n*** {}".format(self.literal))
        PgNode.show(self)
    
    
    def __eq__(self, other):
        '''
        Equality test for nodes - compares only the literal for equality
        
        :param other: PgNode_s
        :return: bool
        '''
        if not isinstance(other, self.__class__): return False
        return (self.symbol == other.symbol) and (self.is_pos == other.is_pos)
    
    
    def __hash__(self):
        return hash(self.symbol) ^ hash(self.is_pos)
    
    
    def __repr__(self):
        return str(self.symbol) if self.is_pos else "~"+str(self.symbol)



class PgNode_a(PgNode):
    '''
    A-type (action) Planning Graph node - inherited from PgNode
    '''

    def __init__(self, action: Action):
        '''
        A-level Planning Graph node constructor
        
        :param action: Action
            a ground action, i.e. this action cannot contain any variables
        Instance variables calculated:
            An A-level will always have an S-level as its parent and an S-level as its child.
            The preconditions and effects will become the parents and children of the A-level node
            However, when this node is created, it is not yet connected to the graph
            prenodes: set of *possible* parent S-nodes
            effnodes: set of *possible* child S-nodes
            is_persistent: bool   True if this is a persistence action, i.e. a no-op action
        Instance variables inherited from PgNode:
            parents: set of nodes connected to this node in previous S level; initially empty
            children: set of nodes connected to this node in next S level; initially empty
            mutex: set of sibling A-nodes that this node has mutual exclusion with; initially empty
       '''
        PgNode.__init__(self)
        self.action = action
        self.is_persistent = (
            action.precond_pos == action.effect_add and
            action.precond_neg == action.effect_rem
        )
    
    
    def show(self):
        '''
        Helper print for debugging shows action plus counts of parents, children, siblings
        
        :return:
            print only
        '''
        print("\n*** {}{}".format(self.action.name, self.action.args))
        PgNode.show(self)
    

    def __eq__(self, other):
        '''
        Equality test for nodes - compares only the action name for equality
        
        :param other: PgNode_a
        :return: bool
        '''
        if isinstance(other, self.__class__):
            return (self.action.name == other.action.name) \
                   and (self.action.args == other.action.args)

    def __hash__(self):
        return hash(self.action.name) ^ hash(self.action.args)
    
    
    def __repr__(self):
        return self.action.name+str(self.action.args)



def mutexify(node1: PgNode, node2: PgNode):
    '''
    Adds sibling nodes to each other's mutual exclusion (mutex) set. These should be sibling nodes!
    
    :param node1: PgNode (or inherited PgNode_a, PgNode_s types)
    :param node2: PgNode (or inherited PgNode_a, PgNode_s types)
    :return:
        node mutex sets modified
    '''
    if type(node1) != type(node2):
        raise TypeError('Attempted to mutex two nodes of different types')
    node1.mutex.add(node2)
    node2.mutex.add(node1)



class PlanningGraph():
    '''
    A planning graph as described in chapter 10 of the AIMA text. The planning
    graph can be used to reason about the actions most likely to meet goal-
    conditions by working forwards from the current state through layers of
    actions and literals that loosely correspond to the space of possible
    outcomes.
    '''
    
    def __init__(self, problem: Problem, state: str, serial_planning=True):
        '''
        :param problem: PlanningProblem (or subclass such as AirCargoProblem or HaveCakeProblem)
        :param state: str (will be in form TFTTFF... representing fluent states)
        :param serial_planning: bool (whether or not to assume that only one action can occur at a time)
        Instance variable calculated:
            fs: FluentState
                the state represented as positive and negative fluent literal lists
            all_actions: list of the PlanningProblem valid ground actions combined with calculated no-op actions
            s_levels: list of sets of PgNode_s, where each set in the list represents an S-level in the planning graph
            a_levels: list of sets of PgNode_a, where each set in the list represents an A-level in the planning graph
        '''
        self.problem = problem
        self.fs = decode_state(state, problem.state_map)
        self.serial = serial_planning
        self.all_actions = self.problem.actions_list + self.noop_actions(self.problem.state_map)
        self.s_levels = []
        self.a_levels = []
        self.create_graph()
    
    
    def print_graph(self):
        '''
        Prints the current state of the graph.
        '''
        print("\nPRINTING PLANNING GRAPH")
        print("  S Level 0", ":", [n for n in self.s_levels[0]])
        for level in range(len(self.a_levels)):
            a_level = self.a_levels[level]
            s_level = self.s_levels[level + 1]
            print("  A Level", level    , ":", a_level)
            print("  S Level", level + 1, ":", s_level)
    
    
    def noop_actions(self, literal_list):
        '''
        Creates a persistent action for each possible fluent
        
        "No-Op" actions are virtual actions (i.e., actions that only exist in
        the planning graph, not in the planning problem domain) that operate
        on each fluent (literal expression) from the problem domain. No op
        actions "pass through" the literal expressions from one level of the
        planning graph to the next.
        
        The no-op action list requires both a positive and a negative action
        for each literal expression. Positive no-op actions require the literal
        as a positive precondition and add the literal expression as an effect
        in the output, and negative no-op actions require the literal as a
        negative precondition and remove the literal expression as an effect in
        the output.
        
        This function should only be called by the class constructor.
        
        :param literal_list:
        :return: list of Action
        '''
        action_list = []
        for fluent in literal_list:
            act1 = Action(expr("Noop_pos({})".format(fluent)), ([fluent], []), ([fluent], []))
            action_list.append(act1)
            act2 = Action(expr("Noop_neg({})".format(fluent)), ([], [fluent]), ([], [fluent]))
            action_list.append(act2)
        return action_list
    
    
    def create_graph(self):
        '''
        Builds a Planning Graph as described in Russell-Norvig 3rd Ed 10.3 or 2nd Ed 11.4
        
        The S0 initial level has been implemented for you.  It has no parents and includes all of
        the literal fluents that are part of the initial state passed to the constructor.  At the start
        of a problem planning search, this will be the same as the initial state of the problem.  However,
        the planning graph can be built from any state in the Planning Problem
        
        This function should only be called by the class constructor.
        
        :return:
            builds the graph by filling s_levels[] and a_levels[] lists with node sets for each level
        '''
        # the graph should only be built during class construction
        if (len(self.s_levels) != 0) or (len(self.a_levels) != 0):
            raise Exception(
                'Planning Graph already created; construct a new planning graph for each new state in the planning sequence')
        
        # initialize S0 to literals in initial state provided.
        leveled = False
        level = 0
        self.s_levels.append(set())  # S0 set of s_nodes - empty to start
        # for each fluent in the initial state, add the correct literal PgNode_s
        for literal in self.fs.pos:
            self.s_levels[level].add(PgNode_s(literal, True))
        for literal in self.fs.neg:
            self.s_levels[level].add(PgNode_s(literal, False))
        # no mutexes at the first level
        
        # continue to build the graph alternating A, S levels until last two S levels contain the same literals,
        # i.e. until it is "leveled"
        while not leveled:
            self.add_action_level(level)
            self.update_a_mutex(self.a_levels[level])
            
            level += 1
            self.add_literal_level(level)
            self.update_s_mutex(self.s_levels[level])
            
            if self.s_levels[level] == self.s_levels[level - 1]:
                leveled = True
    
    
    def add_action_level(self, level):
        '''
        Adds an A (action) level to the Planning Graph
        
        :param level: int
            the level number alternates S0, A0, S1, A1, S2, .... etc the level number is also used as the
            index for the node set lists self.a_levels[] and self.s_levels[]
        :return:
            adds A nodes to the current level in self.a_levels[level]
        '''
        s_level = self.s_levels[level]
        a_level = []
        for action in self.all_actions:
            pos_parents = [node for node in s_level if node.symbol in action.precond_pos and     node.is_pos]
            neg_parents = [node for node in s_level if node.symbol in action.precond_neg and not node.is_pos]
            if not (pos_parents or neg_parents): continue
            
            a_node = PgNode_a(action)
            for s_node in (neg_parents + pos_parents):
                a_node.parents.add(s_node)
                s_node.children.add(a_node)
            
            a_level.append(a_node)
        
        self.a_levels.append(a_level)
    
    
    def add_literal_level(self, level):
        '''
        Adds an S (literal) level to the Planning Graph
        
        :param level: int
            the level number alternates S0, A0, S1, A1, S2, .... etc the level number is also used as the
            index for the node set lists self.a_levels[] and self.s_levels[]
        :return:
            adds S nodes to the current level in self.s_levels[level]
        '''
        
        a_level = self.a_levels[level - 1]
        pos_literals = set()
        neg_literals = set()
        for node in a_level:
            for effect in node.action.effect_add: pos_literals.add(effect)
            for effect in node.action.effect_rem: neg_literals.add(effect)
        
        s_level = []
        for l in pos_literals: s_level.append(PgNode_s(l, True ))
        for l in neg_literals: s_level.append(PgNode_s(l, False))
        
        for s_node in s_level:
            parents = None
            if s_node.is_pos: parents = [a for a in a_level if s_node.symbol in a.action.effect_add]
            else:             parents = [a for a in a_level if s_node.symbol in a.action.effect_rem]
            for a_node in parents:
                a_node.children.add(s_node)
                s_node.parents.add(a_node)
        
        self.s_levels.append(s_level)
    
    
    def update_a_mutex(self, nodeset):
        '''
        Determine and update sibling mutual exclusion for A-level nodes
        
        Mutex action tests section from 3rd Ed. 10.3 or 2nd Ed. 11.4
        A mutex relation holds between two actions a given level
        if the planning graph is a serial planning graph and the pair are nonpersistence actions
        or if any of the three conditions hold between the pair:
           Inconsistent Effects
           Interference
           Competing needs
        
        :param nodeset: set of PgNode_a (siblings in the same level)
        :return:
            mutex set in each PgNode_a in the set is appropriately updated
        '''
        nodelist = list(nodeset)
        for i, n1 in enumerate(nodelist[:-1]):
            for n2 in nodelist[i + 1:]:
                if (self.serialize_actions(n1, n2) or
                    self.inconsistent_effects_mutex(n1, n2) or
                    self.interference_mutex(n1, n2) or
                    self.competing_needs_mutex(n1, n2)
                ): mutexify(n1, n2)
    
    
    def serialize_actions(self, node_a1: PgNode_a, node_a2: PgNode_a) -> bool:
        '''
        Test a pair of actions for mutual exclusion, returning True if the
        planning graph is serial, and if either action is persistent; otherwise
        return False.  Two serial actions are mutually exclusive if they are
        both non-persistent.
        
        :param node_a1: PgNode_a
        :param node_a2: PgNode_a
        :return: bool
        '''
        if not self.serial:
            return False
        if node_a1.is_persistent or node_a2.is_persistent:
            return False
        return True
    
    
    def inconsistent_effects_mutex(self, a1: PgNode_a, a2: PgNode_a) -> bool:
        '''
        Test a pair of actions for inconsistent effects, returning True if
        one action negates an effect of the other, and False otherwise.
        
        :param node_a1: PgNode_a
        :param node_a2: PgNode_a
        :return: bool
        '''
        matches_1 = [e for e in a1.action.effect_add if e in a2.action.effect_rem]
        matches_2 = [e for e in a2.action.effect_add if e in a1.action.effect_rem]
        return matches_1 or matches_2
    
    
    def interference_mutex(self, a1: PgNode_a, a2: PgNode_a) -> bool:
        '''
        Test a pair of actions for mutual exclusion, returning True if the 
        effect of one action is the negation of a precondition of the other.
        
        :param node_a1: PgNode_a
        :param node_a2: PgNode_a
        :return: bool
        '''
        p_matches_1 = [e for e in a1.action.effect_add if e in a2.action.precond_neg]
        p_matches_2 = [e for e in a2.action.effect_add if e in a1.action.precond_neg]
        n_matches_1 = [e for e in a1.action.effect_rem if e in a2.action.precond_pos]
        n_matches_2 = [e for e in a2.action.effect_rem if e in a1.action.precond_pos]
        return p_matches_1 or p_matches_2 or n_matches_1 or n_matches_2
    
    
    def competing_needs_mutex(self, a1: PgNode_a, a2: PgNode_a) -> bool:
        '''
        Test a pair of actions for mutual exclusion, returning True if one of
        the precondition of one action is mutex with a precondition of the
        other action.
        
        :param node_a1: PgNode_a
        :param node_a2: PgNode_a
        :return: bool
        '''
        for p1 in a1.parents:
            for p2 in a2.parents:
                if p1.is_mutex(p2): return True
        return False
    
    
    def update_s_mutex(self, nodeset: set):
        '''
        Determine and update sibling mutual exclusion for S-level nodes
        
        Mutex action tests section from 3rd Ed. 10.3 or 2nd Ed. 11.4
        A mutex relation holds between literals at a given level
        if either of the two conditions hold between the pair:
           Negation
           Inconsistent support
        
        :param nodeset: set of PgNode_a (siblings in the same level)
        :return:
            mutex set in each PgNode_a in the set is appropriately updated
        '''
        nodelist = list(nodeset)
        for i, n1 in enumerate(nodelist[:-1]):
            for n2 in nodelist[i + 1:]:
                if self.negation_mutex(n1, n2) or self.inconsistent_support_mutex(n1, n2):
                    mutexify(n1, n2)
    
    
    def negation_mutex(self, s1: PgNode_s, s2: PgNode_s) -> bool:
        '''
        Test a pair of state literals for mutual exclusion, returning True if
        one node is the negation of the other, and False otherwise.
        
        :param node_s1: PgNode_s
        :param node_s2: PgNode_s
        :return: bool
        '''
        return s1.symbol == s2.symbol and s1.is_pos != s2.is_pos
    
    
    def inconsistent_support_mutex(self, s1: PgNode_s, s2: PgNode_s):
        '''
        Test a pair of state literals for mutual exclusion, returning True if
        there are no actions that could achieve the two literals at the same
        time, and False otherwise.  In other words, the two literal nodes are
        mutex if all of the actions that could achieve the first literal node
        are pairwise mutually exclusive with all of the actions that could
        achieve the second literal node.
        
        :param node_s1: PgNode_s
        :param node_s2: PgNode_s
        :return: bool
        '''
        for p1 in s1.parents:
            for p2 in s2.parents:
                if not p1.is_mutex(p2): return False
        return True
    
    
    def h_levelsum(self) -> int:
        '''
        The sum of the level costs of the individual goals (admissible if goals independent)
        
        :return: int if all sub-goals are (plausibly) satisfiable, or False if
                 one or more are absent from the graph.
        '''
        level_sum = 0
        for clause in self.problem.goal:
            level = 0
            for s_level in self.s_levels:
                match = [node for node in s_level if node.symbol == clause and node.is_pos]
                if match: break
                level += 1
            if level == len(self.s_levels):
                return False
            level_sum += level
        
        return level_sum
    
    
    def h_setlevel(self) -> int:
        level = 0
        for s_level in self.s_levels:
            
            matches = [n for n in s_level if n.symbol in self.problem.goal and n.is_pos]
            if len(matches) != len(self.problem.goal): continue
            
            match_okay = True
            mutex = ()
            for node in matches:
                mutex = mutex + node.mutex
                if node in mutex:
                    match_okay = False
                    break
            
            if match_okay:
                return level
            
            level += 1
        return False



class ReverseLevelSumLookup():
    '''
    A loose 'inversion' of the Planning Graph, based on working backwards from
    a given problem's goals rather than working forward, and dispensing with
    the assorted mutex checks, parent links and action-levels.  This simplifies
    implementation and allows the structure to be initialised and cached once
    per search, rather than having to be re-built for every state explored.
    
    :param problem: PlanningProblem (or subclass such as AirCargoProblem or HaveCakeProblem)
    '''
    def __init__(self, problem: Problem):
        self.problem = problem
        self.verbose = False
        self.create_graph()
    
    
    def create_graph(self):
        '''
        Creates the 'graph' (really just a sequence of literal levels), derived
        by successively looking for actions which could produce the 'needs'
        on the prior level, and adding their positive preconditions to the
        next level of 'needs'.
        
        As a further refinement, needs on each level are 'weighted' more
        heavily if they contribute to fulfilling multiple needs on the last
        level (and penalised if they threaten them.)  This is used to modify
        the cost estimate produced by h_levelsum (below.)
        '''
        self.need_levels = [{}]
        self.acts_levels = [{}]
        init_needs = self.need_levels[0]
        
        #  We initialise the first level of needs directly from the problem's
        #  goal-state, with a weighting of 1.
        for goal in self.problem.goal:
            init_needs[goal] = 1.
        
        while True:
            
            #  We initialise each new level with the contents of the last (to
            #  ensure that the succession of needs will eventually 'stabilise'-
            #  see below.)
            needs = self.need_levels[-1]
            new_needs = needs.copy()
            new_acts = {}
            max_needing  = 0
            
            #  Each potential action is rated positively for each need it can
            #  satisfy, and penalised for each need it would threaten, both in
            #  proportion to the weighting of that need.
            for action in self.problem.actions_list:
                sum_meets = 0
                sum_takes = 0
                
                for clause in action.effect_add:
                    if clause in needs: sum_meets += needs[clause]
                for clause in action.effect_rem:
                    if clause in needs: sum_takes += needs[clause]
                
                if sum_meets == 0: continue
                action_rating = sum_meets / (1 + sum_takes)
                
                #  If that action can contribute, all it's positive
                #  preconditions are added to the set of successor needs.
                #  NOTE:  negative preconditions were not present for any of
                #  the actions used for testing, so their handling has been
                #  ommitted.
                new_acts[action] = action_rating
                for clause in action.precond_pos:
                    if not clause in new_needs: new_needs[clause] = 0
                    new_needs[clause] += action_rating
                    max_needing = max(max_needing, new_needs[clause])
            
            #  If no new needs have been generated, we don't bother adding the
            #  new level, and simply return.
            if len(new_needs) == 0 or new_needs.keys() == needs.keys(): break
            
            #  Otherwise, we scale the weighting of each new need to fit
            #  between 1 and 0, then append the new level to the sequence.
            for clause in new_needs: new_needs[clause] /= max_needing
            self.need_levels.append(new_needs)
            self.acts_levels.append(new_acts)
        
        #  Optionally, we can print out the sequence of weighted need-levels
        #  constructed.  (Actions are stored solely for debugging/review.)
        if self.verbose: self.print_levels()
    
    
    def print_levels(self):
        '''
        Prints the cached need-levels for this heuristic.
        '''
        print("\n\nGoals are:", self.problem.goal)
        print("Final need levels are:")
        for level_ID in range(len(self.need_levels)):
            acts = self.acts_levels[level_ID]
            if len(acts) > 0: print("  Actions Level", level_ID)
            for action in acts:
                act_name = action.name+str(action.args)
                print("    {:20.20} : {:8.4}".format(act_name, str(acts[action])))
            print("  Needs Level", level_ID)
            needs = self.need_levels[level_ID]
            for need in needs:
                print("    {:20.20} : {:8.4}".format(str(need), str(needs[need])))
        print("\n")
    
    
    def h_levelsum(self, state: str):
        '''
        Returns the sum of estimated step-costs to reach the problem's goals,
        given the current set of positive facts within the given state.
        
        :param state: a string representing the true/false values of each fact
                      within th world.
        :return: an int measuring the guesstimated cost to fulfill the
                 problem's goals.
        '''
        level_sum = 0
        truths = [self.problem.state_map[i] for i in range(len(state)) if state[i] == 'T']
        for truth in truths:
            cost = 0
            for needs in self.need_levels:
                
                #  If a fact from the state is present on a given generation of
                #  needs, adjust the cost based on how 'strongly' it's present
                #  and move on the next fact.  Otherwise, move on to the next
                #  level.
                if truth in needs:
                    cost += 1 - needs[truth]
                    break
                cost += 1
            level_sum += cost
        return level_sum
    
    
    def h_setlevel(self, state: str):
        level = 0
        truths = [self.problem.state_map[i] for i in range(len(state)) if state[i] == 'T']
        for needs in self.need_levels:
            all_weights = [needs[t] for t in truths if t in needs]
            if len(all_weights) == len(truths):
                level += 1 - (sum(all_weights) / len(all_weights))
                break
            level += 1
        return level
            

