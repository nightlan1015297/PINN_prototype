import  tensorflow as tf
import  numpy as np
import  os
from typing import Tuple, List, Union, Callable


def create_history_dict() -> dict:
    """
    Creates a history dictionary.
    """
    return {
        "LOSS_TOTAL": [],
        "LOSS_INITIAL": [],
        "LOSS_RESIDUAL": [],
    }

class DrivenOscillatorPINN(tf.keras.Model):
    """ A class for a driven oscillator PINN model.

    Args:
    """
    def __init__(self, model, residual_weight = 1, initial_weight = 1, **kwargs):
        # Initialize the model
        super().__init__(**kwargs)
        
        # Set model
        
        self.model = model
        
        
        # Define loss function for diffrent error sources
        
        self.residual_loss  = tf.keras.losses.MeanSquaredError()
        self.initial_loss   = tf.keras.losses.MeanSquaredError()
        
        # Define loss weights for diffrent error sources
        
        self.residual_weight  = tf.Variable(residual_weight, trainable=False, name="RESIDUAL_WEIGHT" , dtype=tf.float32)
        self.initial_weight   = tf.Variable(initial_weight,  trainable=False, name="INITIAL_WEIGHT"  , dtype=tf.float32)

        self.residual_loss_trac  = tf.keras.metrics.Mean(name="LOSS_RESIDUAL")
        self.initial_loss_trac   = tf.keras.metrics.Mean(name="LOSS_INITIAL")
        self.total_loss_trac     = tf.keras.metrics.Mean(name="LOSS_TOTAL")
        
        self.initial_condition = tf.Variable([0,0], trainable=False, name="INITIAL_CONDITION", dtype=tf.float32)

        self.gamma   = 0.3
        self.omega_o = 10
        self.omega_d = 15
        
        self.rhs_function = lambda t: tf.sin(self.omega_d*t)
    
    def set_weight(self, residual_weight:float, initial_weight:float ):
        self.residual_weight.assign(residual_weight)
        self.initial_weight.assign(initial_weight)
        
    def set_rhs_function(self, rhs_function):
        self.rhs_function.assign(rhs_function)
    
    def set_physical_parmeters(self, gamma, omega_o, omega_d):
        self.gamma.assign(gamma)
        self.omega_o.assign(omega_o)
        self.omega_d.assign(omega_d)
    
    def set_initial_condition(self, position , velocity):
        self.initial_condition.assign([position,velocity])
        
    @property
    def metrics(self):
        '''
        Returns the metrics of the model.
        '''
        return [self.total_loss_trac, self.residual_loss_trac, self.initial_loss_trac]
    
    
    """
    Call function for the model
    """
    @tf.function
    def call(self, t, training: bool = False):
        initial_t = tf.constant(0,dtype=tf.float32)
        with tf.GradientTape(watch_accessed_variables=False) as gg:
            gg.watch(t)
            with tf.GradientTape(watch_accessed_variables=False) as g:
                g.watch(t)
                g.watch(initial_t)
                u = self.model(t , training=training)
                initial_u = self.model(initial_t , training=training)
            du_dt = g.gradient(u, t)
            dinitial_u_dt = g.gradient(initial_u, initial_t)
        d2u_dt2 = gg.gradient(du_dt, t)
        
        
        lhs_output = d2u_dt2 + self.gamma*du_dt + self.omega_o**2*u

        return u, lhs_output , tf.Variable([initial_u , dinitial_u_dt])

    
    def training_step(self, t):
        with tf.GradientTape() as tape:
            u ,lhs_output , initial_output = self(t, training=True)
            loss_residual = self.residual_loss(lhs_output, self.rhs_function(t))
            loss_initial = self.initial_loss(initial_output, self.initial_condition)
            loss = self.residual_weight*loss_residual + self.initial_weight*loss_initial
        
        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        
        self.residual_loss_trac.update_state(loss_residual)
        self.initial_loss_trac.update_state(loss_initial)
        self.total_loss_trac.update_state(loss)
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, t):
        u ,lhs_output ,initial_output= self(t, training=True)
        loss_residual = self.residual_loss(lhs_output, self.rhs_function(t))
        loss_initial = self.initial_loss(initial_output, self.initial_condition)
        loss = self.residual_weight * loss_residual + self.initial_weight * loss_initial
        
        return {m.name: m.result() for m in self.metrics}
    
    
    def fit_it (self,inputs: List['tf.Tensor'],epochs: int ,batch_size: int ,print_every: int = 1000):
        history = create_history_dict()
        
        for epoch in range(epochs):
            for i in range(0, len(inputs), batch_size):
                if i + batch_size > len(inputs):
                    batch_inputs = inputs[i:]
                else:
                    batch_inputs = inputs[i:i+batch_size]
                metrs = self.training_step(batch_inputs)
        
            for key, value in metrs.items():
                history[key].append(value.numpy())
            
            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {self.total_loss_trac.result()}, Residual Loss: {self.residual_loss_trac.result()}, Initial Loss: {self.initial_loss_trac.result()}")

            for m in self.metrics:
                m.reset_states()
        
        return history