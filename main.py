import numpy as np

n_rows, n_cols = 3, 3
actions = ['Up', 'Down', 'Right', 'Left']

p_correct = 0.8  # probability that the agent goes in the direction it selects
p_wrong = 0.1  # probability it moves at right angles to the intended direction

gamma = 0.99  # Discount factor


def get_new_state(i, j, action):
    if action == 'Up':
        return max(i - 1, 0), j
    elif action == 'Down':
        return min(i + 1, n_rows - 1), j
    elif action == 'Right':
        return i, min(j + 1, n_cols - 1)
    elif action == 'Left':
        return i, max(j - 1, 0)


def value_iteration(r):
    rewards = np.array([[r, -1, 10], [-1, -1, -1], [-1, -1, -1]])
    V = np.zeros((n_rows, n_cols))

    # Terminal states' values are equal to their immediate rewards.
    V[0, 0] = r
    V[0, 2] = 10

    epsilon = 1e-6  # Convergence threshold

    while True:
        delta = 0
        for i in range(n_rows):
            for j in range(n_cols):
                # Skip terminal states
                if (i == 0 and j == 0) or (i == 0 and j == 2):
                    continue

                v = V[i, j]
                max_value = -np.inf

                for action in actions:
                    total_expected_reward = 0
                    # Intended direction
                    new_i, new_j = get_new_state(i, j, action)
                    if i == new_i and j == new_j:
                        continue
                    total_expected_reward += p_correct * (rewards[new_i][new_j] + gamma * V[new_i][new_j])
                    # Right angles to the intended direction
                    for side_action in (
                            actions[(actions.index(action) + 1) % 4], actions[(actions.index(action) - 1) % 4]):
                        new_i, new_j = get_new_state(i, j, side_action)
                        if i == new_i and j == new_j:
                            continue
                        total_expected_reward += p_wrong * (rewards[new_i][new_j] + gamma * V[new_i][new_j])

                    max_value = max(total_expected_reward, max_value)

                V[i][j] = max_value
                delta = max(delta, np.abs(v - V[i][j]))

        if delta < epsilon:
            break

    policy = np.zeros((n_rows, n_cols), dtype=object)
    for i in range(n_rows):
        for j in range(n_cols):
            if (i == 0 and j == 0) or (i == 0 and j == 2):
                policy[i][j] = 'Terminal'
                continue

            max_value = -np.inf
            for action in actions:
                total_expected_reward = 0
                # Intended direction
                new_i, new_j = get_new_state(i, j, action)
                if i == new_i and j == new_j:
                    continue
                total_expected_reward += p_correct * (rewards[new_i][new_j] + gamma * V[new_i][new_j])

                # Right angles to the intended direction
                for side_action in (actions[(actions.index(action) + 1) % 4], actions[(actions.index(action) - 1) % 4]):
                    new_i, new_j = get_new_state(i, j, side_action)
                    if i == new_i and j == new_j:
                        continue
                    total_expected_reward += p_wrong * (rewards[new_i][new_j] + gamma * V[new_i][new_j])

                if total_expected_reward > max_value:
                    max_value = total_expected_reward
                    policy[i][j] = action

    return V, policy


r_values = [100, 3, 0, -3]

def calculate_state_value(i, j, action, V, rewards):
    # Calculate value for moving in intended direction
    new_i, new_j = get_new_state(i, j, action)

    value = p_correct * (rewards[new_i, new_j] + gamma * V[new_i, new_j])

    if i == new_i and j == new_j:
        value = 0
    # Calculate value for moving at right angles
    for side_action in [actions[(actions.index(action) - 1) % 4], actions[(actions.index(action) + 1) % 4]]:
        side_i, side_j = get_new_state(i, j, side_action)
        if i == side_i and j == side_j:
            continue
        value += p_wrong * (rewards[side_i, side_j] + gamma * V[side_i, side_j])

    return value

def policy_evaluation(policy, V, rewards):
    epsilon = 1e-6
    while True:
        delta = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if policy[i][j] == 'Terminal':
                    continue
                v = V[i, j]

                # Get the action suggested by the current policy
                action = policy[i][j]
                new_value = 0

                new_value += calculate_state_value(i, j, action, V, rewards)

                V[i][j] = new_value
                delta = max(delta, abs(v - V[i][j]))

        if delta < epsilon:
            break
    return V


def policy_improvement(V, policy, rewards):
    policy_stable = True
    for i in range(n_rows):
        for j in range(n_cols):
            if policy[i][j] == 'Terminal':
                continue
            old_action = policy[i][j]
            max_value = -np.inf
            for action in actions:
                action_value = calculate_state_value(i, j, action, V, rewards)
                if action_value > max_value:
                    max_value = action_value
                    policy[i][j] = action
            if old_action != policy[i][j]:
                policy_stable = False
    return policy, policy_stable


def policy_iteration(r):
    rewards = np.array([[r, -1, 10], [-1, -1, -1], [-1, -1, -1]])
    V = np.zeros((n_rows, n_cols))
    # Random initial policy
    policy = np.empty((n_rows, n_cols), dtype='U10')
    for i in range(n_rows):
        for j in range(n_cols):
            if (i == 0 and j == 0) or (i == 0 and j == 2):
                policy[i][j] = 'Terminal'
            else:
                policy[i][j] = np.random.choice(actions)
    policy[0][0], policy[0][2] = "Terminal", "Terminal"

    while True:
        V = policy_evaluation(policy, V, rewards)
        policy, policy_stable = policy_improvement(V, policy, rewards)
        if policy_stable:
            break

    return V, policy


r_values = [100, 3, 0, -3]

for r in r_values:
    V, policy = value_iteration(r)
    print("//////////  Value     //////////////")
    print(f"Value function for r = {r}:")
    print(V)
    print(f"Policy for r = {r}:")
    print(policy)
    print("//////////   POLICY  //////////\n")
    V, policy = policy_iteration(r)
    print(f"Value function for r = {r}:")
    print(V)
    print(f"Policy for r = {r}:")
    print(policy)
    print("////////////////////////////////////////////////////////////////////////////\n")
