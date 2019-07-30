"""Run the jobs in this file.

Example commands:
python cinjon_jobs.py cims testtube: Run the func jobs on cims using testtube.
python cinjon_jobs.py cims batch: Run the func jobs on cims using sbatch.
"""
import sys

from run_on_cluster import cims_run_batch, hpc_run_batch, fb_run_batch
from local_config import email, code_directory

# Location is either hpc, cims, or fb.
location = sys.argv[1]

if location == 'hpc':
    func = hpc_run_batch
elif location == 'cims':
    func = cims_run_batch
elif location == 'fb':
    func = fb_run_batch

counter = 0


job = {
    'name': '2019.7.22.tc',
    'manualSeed': 1,
    'dataset': 'vlog',
}
for epochs in [6, 30]:
    for num_gpus in [4, 8]:
        for batchSize in [9, 12]:
            for num_run in range(2):
                counter += 1

                _job = {k: v for k, v in job.items()}
                effBatchSize = batchSize * num_gpus
                _job['batchSize'] = effBatchSize
                _job['lr'] = 2e-4 * float(effBatchSize) / 36.
                _job['name'] = '%s-%d.%d.%s' % (_job['name'], counter,
                                                num_run, location)
                _job['epochs'] = epochs

                _job['num_gpus'] = num_gpus
                if location == 'fb':
                    _job['num_cpus'] = num_gpus * 10
                    _job['gb'] = 64 * num_gpus
                else:
                    _job['num_cpus'] = 16
                    _job['gb'] = 16 * num_gpus

                _job['time'] = max(epochs, 1)
                # func(_job, counter, email, code_directory)


job = {
    'name': '2019.7.23.tc',
    'manualSeed': 1,
    'dataset': 'vlog',
}
for epochs in [50]:
    for num_gpus in [4, 8]:
        for batchSize in [9, 12, 15]:
            counter += 1

            _job = {k: v for k, v in job.items()}
            effBatchSize = batchSize * num_gpus
            _job['batchSize'] = effBatchSize
            _job['lr'] = 2e-4 * float(effBatchSize) / 36.
            _job['name'] = '%s-%d.%s' % (_job['name'], counter, location)
            _job['epochs'] = epochs
            
            _job['num_gpus'] = num_gpus
            if location == 'fb':
                _job['num_cpus'] = num_gpus * 10
                _job['gb'] = 64 * num_gpus
            else:
                _job['num_cpus'] = 16
                _job['gb'] = 16 * num_gpus
                
            _job['time'] = max(epochs // 2, 1)
            func(_job, counter, email, code_directory)
                
