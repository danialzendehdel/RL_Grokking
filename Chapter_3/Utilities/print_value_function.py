import numpy as np

def print_state_value_function(V, grid_width):
    length = V.shape[0]
    rows = length // grid_width

    for s in range(rows):
        print("\n" + " ".join(f"{s*grid_width + a:2d}, {V[s*grid_width + a]:6.2f} |" for a in range(grid_width)))
    print("\n")


def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

def print_state_action_values(Q, V, grid_width):
    length = V.shape[0]
    rows = length // grid_width

    for s in range(rows):
        # Print top action values with extra spacing
        top_row = ""
        for a in range(grid_width):
            state_index = s * grid_width + a
            action_values = Q[state_index]
            max_action_index = np.argmax(action_values)
            top_action = f"{action_values[3]:^10.2f}" if max_action_index != 3 else f"*{action_values[3]:^8.2f}*"
            top_row += f"    {top_action:^10}    |"
        print(top_row.rstrip() + "|")

        # Print left action value, state value, and right action value
        middle_row = ""
        for a in range(grid_width):
            state_index = s * grid_width + a
            action_values = Q[state_index]
            max_action_index = np.argmax(action_values)
            left_action = f"{action_values[0]:^6.2f}" if max_action_index != 0 else f"*{action_values[0]:^6.2f}*"
            right_action = f"{action_values[2]:^6.2f}" if max_action_index != 2 else f"*{action_values[2]:^6.2f}*"
            state_value = f"{V[state_index]:^6.2f}"
            middle_row += f" {left_action}  {state_value}  {right_action} |"
        print(middle_row.rstrip() + "|")

        # Print bottom action values
        bottom_row = ""
        for a in range(grid_width):
            state_index = s * grid_width + a
            action_values = Q[state_index]
            max_action_index = np.argmax(action_values)
            bottom_action = f"{action_values[1]:^10.2f}" if max_action_index != 1 else f"*{action_values[1]:^8.2f}*"
            bottom_row += f"    {bottom_action:^10}    |"
        print(bottom_row.rstrip() + "|")

        print("\n")
    print("\n")