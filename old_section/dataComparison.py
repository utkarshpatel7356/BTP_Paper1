
import gurobipy as gp
from gurobipy import GRB
import numpy as np

def minimax_envy_optimization(n, k, v, weights):
    """
    Solve the minimax envy allocation problem using a more direct formulation.
    
    Parameters:
    - n: number of agents
    - k: number of supervisors
    - v: list/array of supervisor values
    - weights: 2D array of preference-based weights
    
    Returns:
    - Optimal allocation, utilities, and maximum envy
    """
    
    # Create the Gurobi model
    model = gp.Model("MinimaxEnvyAllocation")
    
    # Decision variables
    x = {}  # Assignment variables
    u = {}  # Utility variables
    
    # Create binary assignment variables
    for i in range(k):
        for j in range(n):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    # Create utility variables for each agent
    for j in range(n):
        u[j] = model.addVar(lb=-GRB.INFINITY, name=f"u_{j}")
    
    # Maximum envy variable
    max_envy = model.addVar(lb=0, name="max_envy")
    
    # Constraint: Each agent is assigned exactly one supervisor
    for j in range(n):
        model.addConstr(gp.quicksum(x[i, j] for i in range(k)) == 1, f"agent_{j}_assignment")
    
    # Constraint: Each supervisor gets at least one agent
    for i in range(k):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) >= 1, f"supervisor_{i}_assignment")
    
    # Utility computation constraints
    for j in range(n):
        # Find the supervisor assigned to agent j
        utility_expr = gp.LinExpr()
        for i in range(k):
            # Compute supervisor's load (total weight of assigned agents)
            supervisor_load = gp.quicksum(weights[i][m] * x[i, m] for m in range(n))
            
            # Utility for agent j if assigned to supervisor i
            utility_term = (weights[i][j] * v[i] * x[i, j]) / supervisor_load
            utility_expr += utility_term
        
        # Set utility variable
        model.addConstr(u[j] == utility_expr, f"utility_{j}")
    
    # Envy constraints
    for j in range(n):
        for m in range(n):
            if j != m:
                # Envy is the max difference in utilities
                model.addConstr(max_envy >= u[m] - u[j], f"envy_{j}_{m}")
    
    # Objective: Minimize maximum envy
    model.setObjective(max_envy, GRB.MINIMIZE)
    
    # Solve the model
    model.optimize()
    
    # Extract results
    if model.status == GRB.OPTIMAL:
        # Create allocation matrix
        allocation = np.zeros((k, n), dtype=int)
        for i in range(k):
            for j in range(n):
                allocation[i, j] = round(x[i, j].X)
        
        # Extract utilities
        utilities = np.zeros(n)
        for j in range(n):
            utilities[j] = u[j].X
        
        max_envy_value = max_envy.X
        
        return allocation, utilities, max_envy_value
    else:
        print(f"Optimization failed with status {model.status}")
        return None, None, None

def generate_example_data(n=10, k=3, gamma=0.9):
    """
    Generate example data for the problem.
    
    Parameters:
    - n: number of agents
    - k: number of supervisors
    - gamma: discount factor for preference-based weights
    """
    # Supervisor values: random values between 6 and 10
    v = np.random.uniform(36, 60, k)
    
    # Generate preference-based weights using the gamma formula
    weights = np.zeros((k, n))
    for i in range(k):
        # Generate a random preference ordering for supervisor i
        pref_order = np.random.permutation(n)
        print(f"Supervisor {i} preference order: {pref_order}")
        
        # Assign weights based on rank in the preference list
        for rank, j in enumerate(pref_order):
            # wi,j = γ^rank_i(j) as per the model formulation
            # rank is 0-indexed, but formula uses 1-indexed, so we add 1
            weights[i, j] = gamma ** (rank + 1)
    
    return n, k, v, weights

def potential_based_envy_minimization(agent_count, reward_count, rewards, weights, gamma):
    """
    Perform potential-based envy minimization.
    
    Parameters:
    - agent_count: number of agents
    - reward_count: number of supervisors/rewards
    - rewards: array of reward values
    - weights: 2D array of preference-based weights
    - gamma: discount factor
    
    Returns:
    - Final assignments
    - Envy value
    - Potential value
    - Minimum utility
    """
    import numpy as np
    
    # Initialize random assignments - each agent gets exactly one supervisor
    assignments = np.zeros(agent_count, dtype=int)
    for i in range(agent_count):
        assignments[i] = np.random.randint(0, reward_count)
    
    # Make sure every supervisor has at least one agent
    for i in range(reward_count):
        if i not in assignments:
            # Assign a random agent to this supervisor
            assignments[np.random.randint(0, agent_count)] = i
    
    # Weight factor calculation
    def get_weight_factor(assignments, agent_number):
        reward_index = assignments[agent_number]
        my_weight = weights[reward_index][agent_number]
        
        sum_weights = sum(
            weights[assignments[i]][i]
            for i in range(len(assignments)) 
            if assignments[i] == reward_index
        )
        
        return my_weight / sum_weights if sum_weights > 0 else 0.0

    # Reward calculation
    def get_reward(assignments, agent_number):
        weight_factor = get_weight_factor(assignments, agent_number)
        reward_value = rewards[assignments[agent_number]]
        return weight_factor * reward_value

    # Envy calculation
    def calculate_envy(weighted_rewards):
        max_envy = 0
        for i in range(agent_count):
            for j in range(agent_count):
                if i != j:
                    envy = max(0, weighted_rewards[j] - weighted_rewards[i])
                    if envy > max_envy:
                        max_envy = envy
        return max_envy

    # Potential function calculation
    def calculate_potential(weighted_rewards):
        epsilon = 1e-10  # To avoid log(0)
        return sum(np.log(np.maximum(weighted_rewards, epsilon)))

    # Function to find the best assignment for an agent
    def shift_an_agent(assignment, agent_number):
        current_weighted_rewards = np.array([get_reward(assignment, i) for i in range(agent_count)])
        current_potential = calculate_potential(current_weighted_rewards)
        
        best_potential = current_potential
        best_assignment = assignment[agent_number]
        
        # Try each possible supervisor
        for candidate in range(reward_count):
            if candidate == assignment[agent_number]:
                continue
                
            temp_assignment = assignment.copy()
            temp_assignment[agent_number] = candidate
            
            temp_weighted_rewards = np.array([get_reward(temp_assignment, i) for i in range(agent_count)])
            candidate_potential = calculate_potential(temp_weighted_rewards)
            
            if candidate_potential > best_potential:
                best_potential = candidate_potential
                best_assignment = candidate
        
        assignment[agent_number] = best_assignment
        return assignment

    # Iteration function - one full round of best responses
    def one_iteration(assignments):
        new_assignments = assignments.copy()
        for i in range(agent_count):
            new_assignments = shift_an_agent(new_assignments, i)
        return new_assignments

    def max_ut_gain(assignment, agent_number):
        """
        Calculate and display the maximum utility gain an agent can achieve
        through a unilateral shift to another supervisor.
        
        Parameters:
        - assignment: current assignment of agents to supervisors
        - agent_number: the agent to evaluate
        
        Returns:
        - None (prints results)
        """
        # Calculate current utility
        current_utility = get_reward(assignment, agent_number)
        current_supervisor = assignment[agent_number]
        
        # Find the maximum utility by trying each supervisor
        max_utility = current_utility
        best_supervisor = current_supervisor
        
        for candidate in range(reward_count):
            if candidate == current_supervisor:
                continue
                
            temp_assignment = assignment.copy()
            temp_assignment[agent_number] = candidate
            
            candidate_utility = get_reward(temp_assignment, agent_number)
            
            if candidate_utility > max_utility:
                max_utility = candidate_utility
                best_supervisor = candidate
        
        # Calculate the potential gain
        utility_gain = max_utility - current_utility
        
        # Print results
        print(f"Agent {agent_number} - Current utility: {current_utility:.4f} with supervisor {current_supervisor}")
        
        if utility_gain > 0:
            print(f"  Max utility: {max_utility:.4f} with supervisor {best_supervisor}")
            print(f"  Potential gain: {utility_gain:.4f}")
        else:
            print(f"  Already at optimal utility (no gain possible)")
        
        return utility_gain, best_supervisor

    # Main optimization loop
    count = 0
    max_iterations = 100  # Prevent infinite loops
    
    print("Starting potential-based optimization...")
    print(f"Initial assignments: {assignments}")
    
    while count < max_iterations:
        old_assignments = assignments.copy()
        assignments = one_iteration(assignments)
        
        # Check if assignments have converged
        if np.array_equal(old_assignments, assignments):
            print(f"Converged after {count+1} iterations")
            break
        
        count += 1
    
    if count == max_iterations:
        print(f"Reached maximum iterations ({max_iterations})")

    # Final weighted rewards and statistics
    weighted_rewards = np.array([get_reward(assignments, i) for i in range(agent_count)])
    envy_value = calculate_envy(weighted_rewards)
    potential_value = calculate_potential(weighted_rewards)
    
    # Print utility values for each agent
    print("\nFinal utilities:")
    min_ut = float('inf')  # Initialize to positive infinity
    for i in range(agent_count):
        print(f"Agent {i} (assigned supervisor {assignments[i]}): Utility = {weighted_rewards[i]:.4f}")
        min_ut = min(min_ut, weighted_rewards[i])

    print("\nMaximum utility gain:")
    max_gains = []
    for i in range(agent_count):
        gain, _ = max_ut_gain(assignments, i)
        max_gains.append(gain)
    
    print(f"\nMaximum potential gain across all agents: {max(max_gains):.4f}")
    
    return assignments, envy_value, potential_value, min_ut

if __name__ == "__main__":
    # Set random seed for reproducibility
    # np.random.seed(42)
    
    # Generate example data with 10 agents, 3 supervisors, and gamma=0.9
    gamma = 0.3

    # Generate example data
    n, k, v, weights = generate_example_data(n=12, k=5, gamma=gamma)

    print(f"gamma = {gamma}")
    # Display information
    print(f"Number of agents: {n}")
    print(f"Number of supervisors: {k}")
    print(f"Supervisor values: {v}")
    print("Preference-based weights:")
    print(weights)
    
    # Compute max and min supervisor values
    v_max = np.max(v)
    v_min = np.min(v)
    
    # Run the optimization
    allocation, utilities, max_envy_value = minimax_envy_optimization(n, k, v, weights)

    # int max_ut=0
    
    if allocation is not None:
        print("\nOptimal allocation:")
        for i in range(k):
            assigned_agents = [j for j in range(n) if allocation[i, j] == 1]
            print(f"Supervisor {i} (value {v[i]:.2f}) assigned to agents: {assigned_agents}")
        
        print("\nAgent utilities:")
        for j in range(n):
            print(f"Agent {j}: {utilities[j]:.4f}")
            # max_ut=max(max_ut,utilities[j])
        
        print(f"\nMaximum envy: {max_envy_value:.4f}")

 # Potential-based Envy Minimization
    print("\n--- Potential-based Envy Minimization ---")
    pot_assignments, pot_envy, pot_potential,min_ut = potential_based_envy_minimization(
        agent_count=n, 
        reward_count=k, 
        rewards=v,
        weights=weights,
        gamma=gamma
    )
    
    print("\nPotential-based Allocation:")
    unique_rewards = np.unique(pot_assignments)
    for r in unique_rewards:
        assigned_agents = np.where(pot_assignments == r)[0]
        print(f"Supervisor {r} (value {v[r]:.2f}): Agents {assigned_agents.tolist()}")
    
    print(f"\nPotential-based Maximum Envy: {pot_envy:.4f}")
    print(f"Potential-based Potential Value: {pot_potential:.4f}")

    # print(min_ut)
    env_bound = min_ut * ((3 / (gamma ** (n - 1))) * (v_max / v_min) - 1)
    print(f"Paper bound on envy = {env_bound:.4f}")
