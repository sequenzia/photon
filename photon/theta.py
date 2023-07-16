import tensorflow as tf


class Theta:

    def __init__(self, gauge):

        self.gauge = gauge
        self.logs_on = self.gauge.log_theta

        self.params = {'model_pre': [],
                       'model_post': [], 'opt': [], 'grads': []}

    def save_params(self, param_type, grads=None):

        if self.logs_on:

            if param_type == 'model_pre' or param_type == 'model_post':

                for idx, p in enumerate(self.gauge.src.trainable_variables):

                    p_name = p.name

                    p_data = {'idx': idx,
                              'name': p_name,
                              'shape': p.shape,
                              'avg': tf.math.reduce_mean(p).numpy(),
                              'value': p.numpy()}

                    self.params[param_type].append(p_data)

                    if grads is not None:

                        if grads[idx] is not None:
                            grads_data = {'idx': idx,
                                          'name': p_name,
                                          'shape': grads[idx].shape,
                                          'avg': tf.math.reduce_mean(grads[idx]).numpy(),
                                          'value': grads[idx].numpy()}

                            self.params['grads'].append(grads_data)

            if param_type == 'opt':

                for idx, opt in enumerate(self.gauge.src.optimizer.get_weights()):

                    if idx > 0:
                        opt_data = {'idx': idx,
                                    'shape': opt.shape,
                                    'avg': tf.math.reduce_mean(opt).numpy(),
                                    'value': opt}

                        self.params['opt'].append(opt_data)

    def log_params(self, epoch_idx):

        if self.logs_on:

            if len(self.gauge.logs.theta) <= epoch_idx:
                self.gauge.logs.theta.append([])

            self.gauge.logs.theta[epoch_idx].append(self.params.copy())

            self.params = {'model_pre': [],
                           'model_post': [], 'opt': [], 'grads': []}
