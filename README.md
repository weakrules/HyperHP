# HyperHP

1. First generate synthetic data:<br /><pre>python gene_syn_data_gmm.py</pre> You can specify the base, impact, decay, and delay effect manually.

2. To train the main model and learn delay, use <pre>python train_syn_data_MLE_hyper_GPU_gmm.py</pre>. The results will be stored in 'log/out'

3. To test and make predictions of event time, use <pre>python test_MLE_CPU.py</pre>
