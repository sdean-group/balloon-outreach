import scipy.interpolate
import scipy.signal
from env.balloon import Balloon
from env.balloon_env import BalloonEnvironment, BalloonERAEnvironment, BaseBalloonEnvironment
from env.ERA_wind_field import WindField
import numpy as np
import jax
import jax.numpy as jnp
import scipy
from functools import partial
import time
import json
import copy

def sigmoid(x):
    """Returns a sigmoid function with range [-1, 1] instead of [0,1]."""
    return 2 / (1 + np.exp(-x)) - 1

def inverse_sigmoid(x):
    """Returns the inverse of the custom sigmoid defined above. """
    return np.log((x+1)/(1-x))

def balloon_cost(balloon: Balloon, target_state):
    state = np.array([balloon.lat, balloon.lon, balloon.alt])
    cost = np.linalg.norm(state- target_state)
    sand_fraction = balloon.sand/balloon.max_sand
    sand_cost = (1 -  (1 / (1 + np.exp(-100*(sand_fraction - 0.1)))))
    return cost + sand_cost

def plan_cost(plan, balloon_env: BaseBalloonEnvironment):
        """Returns the cost of a plan. The cost is calculated as the euclidean distance between the final state after executing the plan and the target state."""
        #For implementing fly-as-far, could have it be the negative distance from the original starting point
        cost = 0.0

        plan = sigmoid(plan)

        curr_balloon_env = copy.deepcopy(balloon_env)
        target_state = np.array([balloon_env.target_lat, balloon_env.target_lon, balloon_env.target_alt])
        init_state = np.array([curr_balloon_env.balloon.lat, curr_balloon_env.balloon.lon, curr_balloon_env.balloon.alt])
        for i in range(len(plan)):
            cost += (0.99**i) * balloon_cost(curr_balloon_env.balloon, target_state)
            action = plan[i]
            # Check if the sand/power gets too low?
            if curr_balloon_env.balloon.sand < 5:
                action = 0.0
            final_state,reward, done, reason = curr_balloon_env.step(action)
            if done:
                print(f"In cost, reason for stopping: {reason}")
                break
            
        #jax.debug.print("final state x = {x}, y = {y}, pressure = {p}", x=final_balloon.state.x, y=final_balloon.state.y, p=final_balloon.state.pressure)
        final_state = np.array(final_state[:3])

        terminal_cost = (0.99**len(plan)) * (balloon_cost(curr_balloon_env.balloon, target_state))
        
        #cost = np.linalg.norm(final_state- target_state)
        cost_deg = -np.dot((final_state - init_state), (target_state - init_state))
        alt_penalty = np.exp(abs(final_state[2] - target_state[2]))
        return cost + terminal_cost


np.random.seed(seed=42)
def get_initial_plans(balloon:Balloon, forecast:WindField, balloon_env:BaseBalloonEnvironment, num_plans, plan_steps, plan_type):
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

#             Step 59: lat: 17.90, lon: -28.03, alt: 15.36

# Episode terminated: No sand left
# Episode finished with total reward: -29788.86
#Step 99: lat: 14.18, lon: -53.61, alt: 25.00
#Episode finished with total reward: -49463.62

            plan = np.zeros((plan_steps, ))
            plan[:steps] = +0.99 if going_up else -0.99 
            # print(random_height, steps)
            try:
                if steps < plan_steps:
                    plan[steps:] += np.random.uniform(-0.3, 0.3, plan_steps - steps)
            except Exception as e:
                print(f'err {e}')
                print(balloon.alt, random_height, steps, plan_steps)

            plans.append(plan)
    return inverse_sigmoid(np.array(plans))


class MPPIAgent:
        
    def __init__(self, balloon_env=None, control_horizon=10): # Sequence[int]
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

        self.mppi_K = 100  # Number of samples
        self.mppi_lambda = 0.25
        self.mppi_sigma = 0.1  # Standard deviation of noise

        self.plan_time = 2*24*60*60
        self.time_delta = 3*60
        self.stride = 10

        # self.plan_steps = 960 + 23 
        self.plan_steps = 100 # from main.py
        # self.N = self.plan_steps

        self.plan = None # jnp.full((self.plan_steps, ), fill_value=1.0/3.0)
        self.i = 0

        self.key = jax.random.key(seed=0)

        self.avg_opt_time=0
        self.curr_mean = 0

        self.balloon = None
        self.time = None
        self.steps_within_radius = 0

    def _deadreckon(self):
        """ Calculates the current position of a moving object by using a previously determined position, and incorporating estimates of speed, heading, and elapsed time.
        An observation is a state. it has balloon_obersvation and wind information. we use ob_t and MPC predicts a_t which then gives us ob_t+1
        """
        balloon:Balloon = self.balloon_env.balloon
        pressure = balloon.altitude_to_pressure(balloon.alt)
        wind_vector = self.wind_field.get_wind(balloon.lon, balloon.lat, pressure, self.balloon_env.current_time)

        # print(self.balloon.state.time_elapsed/3600.0)
        action = self.plan[self.i]
        state, reward, done, reason = self.balloon_env.step(action)

        #could keep track of steps taken towards target/away from init location

    np.random.seed(42)
    def mppi_optimize(self, nominal_plan, mean=0):
        K = self.mppi_K #num of samples

        plans = np.tile(nominal_plan, (K, 1)) + np.random.normal(mean, self.mppi_sigma, size=(K, self.plan_steps))
        
        plans = np.clip(plans, -4, 4)  # covers most of the inverse sigmoid space, should help with stability

        costs = []
        for k in range(K):
            plan_k = plans[k]
            cost_k = plan_cost(plan_k, self.balloon_env)
            print(f'------------------ mppi iter {k} done-------------------')
            costs.append(cost_k)
        # updates mean and variance by averaging plans by combination of cost and noise term (noise is control term)
        # normally use the random noise is derivative of control
        #softmin weighting so each is between 0 or 1
        costs = np.array(costs)
        beta = np.min(costs)
        weights = np.exp((-1 / self.mppi_lambda) * (costs - beta))

        weights /= np.sum(weights)

        #Multiply this with the SGF filter thing? also add U wtf is that
        weighted_plan = np.average(plans, axis=0, weights=weights)
        #Apply SGF filter
        #weighted_plan = scipy.signal.savgol_filter(weighted_plan, int(len(weighted_plan)/2), int(len(weighted_plan)/4))
        print(f"weighted plan is {weighted_plan[:15]}")
        #return weighted_plan, mean
        return weighted_plan
    
    def begin_episode(self, state: np.ndarray, max_steps:int) -> int:        
        #print('USING ' + initialization_type + ' INITIALIZATION')
        initial_plans = get_initial_plans(self.balloon_env.balloon,self.balloon_env.wind_field, self.balloon_env, 10, max_steps, 'best_altitude')
        
            
        batched_cost = []
        for i in range(len(initial_plans)):
            batched_cost.append(plan_cost(initial_plans[i], self.balloon_env))
        
        min_index_so_far = np.argmin(batched_cost)
        min_value_so_far = batched_cost[min_index_so_far]

        initial_plan = initial_plans[min_index_so_far]
 
        if self.plan is not None and plan_cost(self.plan, self.balloon_env) < min_value_so_far:
            print('Using the previous optimized plan as initial plan')
            initial_plan = self.plan
        coast = inverse_sigmoid(np.random.uniform(-0.2, 0.2, size=(max_steps, )))
        if plan_cost(coast, self.balloon_env) < min_value_so_far:
            #print('Using the nothing plan as initial plan')
            initial_plan = coast


        b4 = time.time()       
        start_cost = plan_cost(initial_plan,self.balloon_env)
        plan = self.mppi_optimize(initial_plan)
        # Ensuring actions are between [-1,1]
        plan = sigmoid(plan)
        after_cost = plan_cost(plan, self.balloon_env)
        print(f"MPPI plan, ∆cost = {after_cost} - {start_cost} = {after_cost - start_cost}")
        print(plan[:15])
        self.plan = plan
        after = time.time()
        print(after - b4, 's to get optimized plan')
        

        self.i = 0

        b4 = time.time()
        self._deadreckon()

        action = self.plan[self.i]
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
            initial_plans = get_initial_plans(self.balloon_env.balloon,self.balloon_env.wind_field, self.balloon_env, 10, max_steps, 'constant')
            batched_cost = []
            for i in range(len(initial_plans)):
                batched_cost.append(plan_cost(initial_plans[i], self.balloon_env))
        
            min_index_so_far = np.argmin(batched_cost)

            self.plan = initial_plans[min_index_so_far]
            action = self.plan[self.i]
            # print('action', action)
            return action.item()


        N = min(len(self.plan), self.control_horizon)
        if self.i>0 and self.i%N==0:
            # Padding random actions after the planning horizon
            random_plan = np.random.uniform(-0.3,0.3,(N,))
            self.plan = inverse_sigmoid(np.hstack((self.plan[N:], random_plan)))
            hi = self.begin_episode(state, max_steps)
            return hi
        else:
            # print('not replanning')
            self._deadreckon()
            action = self.plan[self.i]
            # print('action', action)
            return action.item()
            
    
    def end_episode(self, reward: float, terminal: bool = True) -> None:
        self.i = 0
        self.steps_within_radius = 0
        self.balloon = None
        self.plan = None
        # self.plan_steps = 960 + 23

        """
        Step 59: lat: 18.20, lon: -26.23, alt: 15.35
        ADJUSTED COST:
        Step 64: lat: 19.15, lon: -24.33, alt: 16.04

Episode terminated: No sand left
        """
