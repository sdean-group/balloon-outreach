from env.balloon import Balloon
from env.balloon_env import BalloonEnvironment, BalloonERAEnvironment, BaseBalloonEnvironment
from env.ERA_wind_field import WindField
import jax
import numpy as np
from functools import partial
import datetime as dt
import time
import copy
import scipy


def sigmoid(x):
    """Returns a sigmoid function with range [-1, 1] instead of [0,1]."""
    return 2 / (1 + np.exp(-x)) - 1

def inverse_sigmoid(x):
    """Returns the inverse of the custom sigmoid defined above. """
    return np.log((x+1)/(1-x))


def grad_descent_optimizer(initial_plan, balloon_env:BaseBalloonEnvironment, alpha=0.01,max_iters=10, isNormalized=0):
    """Basic gradient descent with just a learning rate and also normalizing the gradient. Doing both the learning rate and normalization seemed to perform worse and take longer. High learning rates had power violations. Slower learning rates performed better than baseline but took more time (almost 4s)."""
    print(f"USING LEARNING RATE {alpha}, MAXITERS {max_iters}, ISNORMALIZED {isNormalized}")
    plan = initial_plan
    alpha = 1
    max_iters = 1

    for gradient_steps in range(max_iters):

        gradient_plan = finite_difference_grad(plan, balloon_env)
        print(f"lol {gradient_plan}")

        if  np.isnan(gradient_plan).any() or abs(np.linalg.norm(gradient_plan)) < 1e-7:
            break
        if isNormalized==1:
            plan -= alpha * gradient_plan / np.linalg.norm(gradient_plan)
        else:
            plan -= alpha * gradient_plan

    return plan, gradient_steps



np.random.seed(seed=42)
def get_initial_plans(num_plans, plan_steps, plan_type):
    """ Returns an initial trajectory for the balloon.

        :param plan_steps: the length of the returned plan
        :param plan_type: the method used to generate the plan. Should be 'random', 'constant' (and later 'tree')
        """

    # should we make `num_plans` plans and average them?
    if plan_type == 'random':
        return np.random.uniform(low=-1.0, high=1.0, size=plan_steps)
    elif plan_type == 'constant':
        return np.zeros(plan_steps)
    # TODO: add tree search below
    else:
        return np.zeros(plan_steps)




def get_initial_plans2(balloon:Balloon, forecast:WindField, balloon_env:BaseBalloonEnvironment, num_plans, plan_steps, plan_type):
    """ Returns an initial trajectory for the balloon.

        :param plan_steps: the length of the returned plan
        :param plan_type: the method used to generate the plan. Should be 'random', 'constant' (and later 'tree')
        """
    print(f'in initial plans, wind field is {forecast}')
    if plan_type == 'best_altitude':

        flight_record = {balloon.alt: 0}

        time_to_top = 0
        max_km_to_explore = 19.1

        up_balloon = balloon
        while time_to_top < plan_steps and up_balloon.alt < max_km_to_explore:
            up_pressure = up_balloon.altitude_to_pressure(up_balloon.alt * 1000)
            wind_vector = forecast.get_wind(up_balloon.lon, up_balloon.lat, up_pressure, balloon_env.current_time)
            up_balloon.step(wind_vector, balloon_env.dt, 0.99)
            time_to_top += 1

            flight_record[up_balloon.alt] = time_to_top

        time_to_bottom = 0
        min_km_to_explore = 15.4

        down_balloon = balloon
        while time_to_bottom < plan_steps and down_balloon.alt > min_km_to_explore:
            down_pressure = down_balloon.altitude_to_pressure(down_balloon.alt * 1000)
            wind_vector = forecast.get_wind(down_balloon.lon, down_balloon.lat, down_pressure, balloon_env.current_time)
            down_balloon.step(wind_vector, balloon_env.dt, -0.99)
            time_to_bottom += 1

            flight_record[down_balloon.alt] = time_to_bottom
        
            # sorted (should be)
            # flight_record = flight_record_down[::-1] + flight_record_up

            # Sort the dictionary by keys (altitudes) and split them into two separate lists
            sorted_flight_record = sorted(flight_record.items())

            flight_record_altitudes = [altitude for altitude, _ in sorted_flight_record]
            flight_record_steps = [steps for _, steps in sorted_flight_record]
            
            interpolator = scipy.interpolate.RegularGridInterpolator((flight_record_altitudes, ), flight_record_steps, bounds_error=False, fill_value=None)

            plans = []

            for i in range(num_plans):
                random_height = np.random.uniform(15.4, 19.1)
                going_up = random_height >= balloon.alt
                steps = int(round(interpolator(np.array([random_height]))[0]))
                steps = max(0, min(steps, plan_steps))
                # print(steps)

                plan = np.zeros((plan_steps, ))
                plan[:steps] = +0.99 if going_up else -0.99 
                # print(random_height, steps)
                try:
                    if steps < plan_steps:
                        plan[steps:] += np.random.uniform(-0.3, 0.3, plan_steps - steps)
                except:
                    print(balloon.alt, random_height, steps, plan_steps)

                plans.append(plan)
        return inverse_sigmoid(np.array(plans))
    


    res = np.array([])
    # Creates multiple plans
    for _ in range(num_plans):
        if plan_type == 'random':
            res = np.append(res, np.random.uniform(low=-1.0, high=1.0, size=plan_steps)) 
        elif plan_type == 'constant':
            res = np.append(res, np.zeros(plan_steps))

        # TODO: add tree search below
        else:
            res = np.append(res, np.zeros(plan_steps))
    return np.reshape(res, (num_plans, plan_steps))

def plan_cost(plan, balloon_env: BaseBalloonEnvironment):
        """Returns the cost of a plan. The cost is calculated as the euclidean distance between the final state after executing the plan and the target state."""
        #For implementing fly-as-far, could have it be the negative distance from the original starting point
        cost = 0.0
        horizon = 10

        plan = sigmoid(plan)
        target_state = np.array([balloon_env.target_lat, balloon_env.target_lon, balloon_env.target_alt])
        curr_balloon_env = copy.deepcopy(balloon_env)
        print('starting cost rollout')
        for i in range(min(len(plan), horizon)):
            action = plan[i]
            final_state,reward, done, reason = curr_balloon_env.step(action)
            cost += np.linalg.norm(np.array(final_state[:3])- target_state)
            if done:
                print(f"In cost, reason for stopping: {reason}")
                cost += 1000.0
                break
        return cost
    
def plan_cost_complex(plan, balloon_env: BaseBalloonEnvironment):
        """Returns the cost of a plan. The cost is calculated as the euclidean distance between the final state after executing the plan and the target state."""
        #For implementing fly-as-far, could have it be the negative distance from the original starting point
        cost = 0.0

        plan = sigmoid(plan)

        curr_balloon_env = copy.deepcopy(balloon_env)
        init_state = np.array([curr_balloon_env.balloon.lat, curr_balloon_env.balloon.lon, curr_balloon_env.balloon.alt])
        for i in range(len(plan)):
            action = plan[i]
            final_state,reward, done, reason = curr_balloon_env.step(action)
            if done:
                print(f"In cost, reason for stopping: {reason}")
                break
        #jax.debug.print("final state x = {x}, y = {y}, pressure = {p}", x=final_balloon.state.x, y=final_balloon.state.y, p=final_balloon.state.pressure)
        final_state = np.array(final_state[:3])
        target_state = np.array([balloon_env.target_lat, balloon_env.target_lon, balloon_env.target_alt])
        cost = np.linalg.norm(final_state- target_state)
        cost_deg = -np.dot((final_state - init_state), (target_state - init_state))
        alt_penalty = np.exp(abs(final_state[2] - target_state[2]))
        return cost

def finite_difference_grad(plan, balloon_env, epsilon=1e-4, sigma=1.0, alpha=1.0):
    # takes a while
    grad = np.zeros_like(plan)
    
    cost_up = np.zeros_like(plan)
    cost_down = np.zeros_like(plan)
    
    
    for i in range(len(plan)):
        if i < 10:
            v = np.random.normal(0,sigma)
            plan_up = plan.copy()
            plan_down = plan.copy()
            plan_up[i] += epsilon*v
            plan_down[i] -= epsilon*v

            cost_up = plan_cost(plan_up, balloon_env)
            cost_down = plan_cost(plan_down, balloon_env)
            grad[i] = ((cost_up - cost_down) *v * alpha) / (2 * epsilon)
            #print(f"costup {cost_up}, costdown{cost_down}, grad{grad[i]}")
        else:
            grad[i] = grad[9]

    return grad

class MPCAgent:
    """
    A Model Predictive Control (MPC) agent that repeatedly forms an optimized plan every 23 timesteps(plan[i] = the action to take at timestep i) to find the optimal path from current state to goal state.

    State: [lat, long, alt]
    Action: continous value [-1, 1] (is -1 descend, 0 stay, 1 ascend?)

    Optimizer: Gradient Descent/Random Search (for no gradient implementation)
    """
        
    def __init__(self, balloon_env=None, control_horizon:int=10): 
        # copied from tree_search_agent.py
        if balloon_env is not None:    # NOTE: should either be BalloonEnvironment or BalloonERAEnvironment
            self.balloon_env:BaseBalloonEnvironment = balloon_env
            self.target_lat:float = balloon_env.target_lat
            self.target_lon:float = balloon_env.target_lon
            self.target_alt:float = balloon_env.target_alt
            self.wind_field:WindField = balloon_env.wind_field
        else:
            print("No environment provided. Initialization failed.")
            return
        self.control_horizon = control_horizon
        #1 step= 1 min
        self.plan_time = 2*24*60*60
        self.time_delta = 3*60

        # self.plan_steps = 960 + 23 
        #self.plan_steps = 240 # (self.plan_time // self.time_delta) // 3
        # self.N = self.plan_steps

        self.plan = None
        self.i = 0
   
        self.steps_within_radius = 0

    def _deadreckon(self):
        """ Calculates the current position of a moving object by using a previously determined position, and incorporating estimates of speed, heading, and elapsed time.
        An observation is a state. it has balloon_obersvation and wind information. we use ob_t and MPC predicts a_t which then gives us ob_t+1
        """
        # print(self.balloon.state.time_elapsed/3600.0)
        action = self.plan[self.i]
        state, reward, done, reason = self.balloon_env.step(action)

        #could keep track of steps taken towards target/away from init location

    def begin_episode(self, state: np.ndarray, max_steps:int) -> int:        
        print('INITIALIZATION')
        initial_plan = get_initial_plans(10, max_steps, 'random')
        min_value_so_far = plan_cost(initial_plan, self.balloon_env)
            
 
        if self.plan is not None and plan_cost(self.plan, self.balloon_env) < min_value_so_far:
            print('Using the previous optimized plan as initial plan')
            initial_plan = self.plan
        coast = inverse_sigmoid(np.random.uniform(-0.2, 0.2, size=(max_steps, )))
        if plan_cost(coast, self.balloon_env) < min_value_so_far:
            #print('Using the nothing plan as initial plan')
            initial_plan = coast


        b4 = time.time()       
        start_cost = plan_cost(initial_plan,self.balloon_env)
        optimized_plan,step= grad_descent_optimizer(initial_plan, self.balloon_env)

        after_cost = plan_cost(optimized_plan, self.balloon_env)
        print(f"Gradient descent optimizer, {step}, ∆cost = {after_cost} - {start_cost} = {after_cost - start_cost}")
        print(optimized_plan[:15])
        # Ensuring actions are between [-1,1]
        optimized_plan = sigmoid(optimized_plan)
        after = time.time()
        print(after - b4, 's to get optimized plan')       

        self.i = 0

        vertical_velocity = copy.deepcopy(self.balloon_env.balloon.vertical_velocity)
        self.plan = np.roll(optimized_plan, -1)
        self.plan[-1] = optimized_plan[0]

        
        self._deadreckon()
        action = np.array([vertical_velocity + optimized_plan[0]])
        #action = optimized_plan[0]
        #print(f"plan is {self.plan}")
        #action = self.plan[self.i]
        return action.item()
    
    def is_goal_state(self, state: np.ndarray, atols: np.ndarray) -> bool:
        """
        Check if the current state is the goal state.

        Args:
            state: Current state of the environment as a numpy array [lat, lon, alt, t]
        
        Returns:
            True if the current state matches the target state within a tolerance, False otherwise.
        """
        return (np.isclose(state[0], self.target_lat, atol=atols[0]) and
                np.isclose(state[1], self.target_lon, atol=atols[1]) and
                np.isclose(state[2], self.target_alt, atol=atols[2]))



    def select_action(self, state: np.ndarray, max_steps:int, step:int) -> int:
        self.i+=1

        lat_long_atol = 1e-2  # 0.01 degrees in latitude/longitude.
        alt_atol = 0.02  # 20 cm in altitude.
        if (self.is_goal_state(state, atols=np.array([lat_long_atol, lat_long_atol, alt_atol]))):
            print(f"reached target state, break main loop")
            return 0.0
        
        if self.plan is None:
            print(f"no plan yet")
            initial_plan = get_initial_plans(10, max_steps, 'random')

            self.plan = initial_plan
            print(f"plan is {self.plan}")
            action = self.plan[self.i]
            # print('action', action)
            return action.item()


        N = min(len(self.plan), self.control_horizon)
        if self.i>0 and self.i%N==0:
            # Padding random actions after the planning horizon
            random_plan = np.random.uniform(-0.3,0.3,(N,))
            self.plan = inverse_sigmoid(np.hstack((self.plan[N:], random_plan)))
            return self.begin_episode(state, max_steps)
        else:
            # print('not replanning')
            self._deadreckon()
            action = np.array([self.balloon_env.balloon.vertical_velocity + self.plan[self.i]])
            action = self.plan[self.i]
            #print(f"plan is {self.plan}")
            # print('action', action)
            return action.item()
 
    def end_episode(self, reward: float, terminal: bool = True) -> None:
        self.i = 0
        self.steps_within_radius = 0
        self.plan = None
        # self.plan_steps = 960 + 23

