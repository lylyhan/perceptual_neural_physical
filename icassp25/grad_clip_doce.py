import numpy as np
import doce

# define the doce environnment

# define the experiment
experiment = doce.Experiment(
  name = 'grad_clip',
  purpose = 'gradient clipping',
  author = 'mathieu Lagrange',
  address = 'mathieu.lagrange@ls2n.fr',
)
# set acces paths (here only storage is needed)
experiment.set_path('output', '/tmp/'+experiment.name+'/', force=True)
experiment._display.export_pdf = 'latex'

# set the plan (factor : modalities)
experiment.add_plan('plan',
  step = ['learn', 'eval'],
  optimizer = ['adam', 'sophia'],
  model = [0, 1, 2, 3, 4, 5, 6, 7],
  loss = ['p', 'pnp'],
  log = [0, 1],
)
# set the metrics
experiment.set_metric(
  name = 'accuracy',
  percent=True,
  higher_the_better= True,
  significance = True,
  precision = 10
  )

experiment.set_metric(
  name = 'acc_std',
  output = 'accuracy',
  func = np.std,
  percent=True
  )


experiment.set_metric(
  name = 'duration',
  path = 'output2',
  lower_the_better= True,
  precision = 2
  )

def step(setting, experiment):
    # the accuracy  is a function of cnn_type, and use of dropout
    accuracy = (len(setting.nn_type)+setting.dropout+np.random.random_sample(experiment.n_cross_validation_folds))/6
    # duration is a function of cnn_type, and n_layers
    duration = len(setting.nn_type)+setting.n_layers+np.random.randn(experiment.n_cross_validation_folds)
    # storage of outputs (the string between _ and .npy must be the name of the metric defined in the set function)
    np.save(experiment.path.output+setting.identifier()+'_accuracy.npy', accuracy)
    np.save(experiment.path.output2+setting.identifier()+'_duration.npy', -duration)

# invoke the command line management of the doce package
if __name__ == "__main__":
  doce.cli.main(experiment = experiment,
                func = step
                )
