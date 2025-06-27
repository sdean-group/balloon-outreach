from env.balloon import Balloon
from env.balloon_env import BalloonEnvironment, BalloonERAEnvironment, BaseBalloonEnvironment
from env.ERA_wind_field import WindField
import jax
import numpy as np
from functools import partial
import datetime as dt
import time
import copy


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

    for gradient_steps in range(max_iters):

        gradient_plan = finite_difference_grad(plan, balloon_env)

        if  np.isnan(gradient_plan).any() or abs(np.linalg.norm(gradient_plan)) < 1e-7:
            # print('Exiting early, |∂plan| =',abs(jnp.linalg.norm(dplan)))
            break
        # print("A", gradient_steps, abs(jnp.linalg.norm(dplan)))
        #plan -= dplan / jnp.linalg.norm(dplan)
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
    res = np.array([])
    for _ in range(num_plans):

        # should we make `num_plans` plans and average them?
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

        plan = sigmoid(plan)

        curr_balloon_env = copy.deepcopy(balloon_env)
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
        return cost
    
@partial(jax.jit, static_argnums=1)
def jax_plan_cost(plan, balloon_env: BaseBalloonEnvironment):
        """ Returns the cost of a plan. The cost is calculated as the *euclidean distance* (change to Haversine?) between the final state after executing the plan and the target state.
        
            :param plan: the series of actions to evaluate
            :param balloon_env: a BaseBalloonEnvironment instance. This can be BalloonEnvironment or            BalloonERAEnvironment
        """
        #For implementing fly-as-far, could have it be the negative distance from the original starting point
        cost = 0.0

        plan = sigmoid(plan)
        

        # Make a new copy of the current balloon environment so stepping through the plan doesn't affect the actual environment
        curr_balloon_env = copy.deepcopy(balloon_env)
        def update_step(i, state):
            action = plan[i]
            # final_state contains the balloon state and the wind column
            final_state, reward, done, reason = curr_balloon_env.step(action)
            if done:
                jax.debug.print(f"In cost, reason for stopping: {reason}")
                i = len(plan)
            return final_state
        
        final_state = jax.lax.fori_loop(0, len(plan), update_step, init_val=np.array([0.0,0.0,0.0]))
        final_state = np.array(final_state[:3])
        target_state = np.array([balloon_env.target_lat, balloon_env.target_lon, balloon_env.target_alt])
        # Euclidean distance cost
        cost = np.linalg.norm(final_state - target_state)
        return cost

def finite_difference_grad(plan, balloon_env, epsilon=1e-5, sigma=1.0, alpha=1.0):
    # takes a while
    #print('in the grad')
    grad = np.zeros_like(plan)
    
    cost_up = np.zeros_like(plan)
    cost_down = np.zeros_like(plan)
    
    
    for i in range(len(plan)):
        if i < 5:
            v = np.random.normal(0,sigma)
            plan_up = plan.copy()
            plan_down = plan.copy()
            plan_up[i] += epsilon*v
            plan_down[i] -= epsilon*v

            cost_up = plan_cost(plan_up, balloon_env)
            cost_down = plan_cost(plan_down, balloon_env)
            grad[i] = ((cost_up - cost_down) *v * alpha) / (2 * epsilon)
        else:
            grad[i] = grad[4]

    return grad

@partial(jax.jit, static_argnums=1)
@partial(jax.grad, argnums=0)
def get_dplan(plan, balloon_env: BaseBalloonEnvironment):
     # jax.debug.print("{balloon}, {wind_field}, {atmosphere}, {terminal_cost_fn}, {time_delta}, {stride}", balloon=balloon, wind_field=wind_field, atmosphere=atmosphere, terminal_cost_fn=terminal_cost_fn, time_delta=time_delta, stride=stride)
    return plan_cost(plan, balloon_env)

class MPCAgent:
    """
    A Model Predictive Control (MPC) agent that repeatedly forms an optimized plan every 23 timesteps(plan[i] = the action to take at timestep i) to find the optimal path from current state to goal state.

    State: [lat, long, alt]
    Action: continous value [-1, 1] (is -1 descend, 0 stay, 1 ascend?)

    Optimizer: Gradient Descent/Random Search (for no gradient implementation)
    """
        
    def __init__(self, balloon_env=None): 
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
        
        #1 step= 30min so 240 steps is 12 hrs
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
        balloon:Balloon = self.balloon_env.balloon
        pressure = balloon.altitude_to_pressure(balloon.alt)
        wind_vector = self.wind_field.get_wind(balloon.lon, balloon.lat, pressure, self.balloon_env.current_time)

        # print(self.balloon.state.time_elapsed/3600.0)
        action = self.plan[self.i]
        state, reward, done, reason = self.balloon_env.step(action)

        #could keep track of steps taken towards target/away from init location

    #should this be a float if we want continous actions??
    def begin_episode(self, state: np.ndarray, max_steps:int) -> int:        
        #print('USING ' + initialization_type + ' INITIALIZATION')
        initial_plans = get_initial_plans(10, max_steps, 'constant')
            
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
        plan,step= grad_descent_optimizer(initial_plan, self.balloon_env)

        after_cost = plan_cost(plan, self.balloon_env)
        print(f"Gradient descent optimizer, {step}, ∆cost = {after_cost} - {start_cost} = {after_cost - start_cost}")
        print(plan[:15])
        # Ensuring actions are between [-1,1]
        self.plan = sigmoid(plan)
        after = time.time()
        print(after - b4, 's to get optimized plan')
        

        self.i = 0

        b4 = time.time()
        self._deadreckon()

        action = self.plan[self.i]
        return action.item()


    def select_action(self, state: np.ndarray, max_steps:int, step:int) -> int:
        self.i+=1
        if self.plan is None:
            initial_plans = get_initial_plans(10, max_steps, 'constant')
            batched_cost = []
            for i in range(len(initial_plans)):
                batched_cost.append(plan_cost(initial_plans[i], self.balloon_env))
        
            min_index_so_far = np.argmin(batched_cost)

            self.plan = initial_plans[min_index_so_far]

        # assuming planning horizon is 23?
        N = min(len(self.plan), 5)
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
        self.plan = None
        # self.plan_steps = 960 + 23

