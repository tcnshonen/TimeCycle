"""Run train on the cluster."""
import os

# NOTE:
# email should be your email. name should be identifying, like `cinjon`.
# code_directory is the directory containing train_video_cycle_simple.py, relative to ${HOME}.
# Examples:
# email = 'cinjon@nyu.edu'
# name = 'cinjon'
# code_directory = 'Code/cycle-consistent-supervision'
# from local_config import email, name, code_directory


def _ensure_dirs(slurm_logs, slurm_scripts):
    for d in [slurm_logs, slurm_scripts]:
        if not os.path.exists(d):
            os.makedirs(d)


def _is_float(v):
    try:
        v = float(v)
        return True
    except:
        return False


def _is_int(v):
    try:
        return int(v) == float(v)
    except:
        return False


def _run_batch(job,
               counter,
               slurm_logs,
               slurm_scripts,
               module_load,
               directory,
               email,
               code_directory,
               local_comet_dir=None):
    _ensure_dirs(slurm_logs, slurm_scripts)

    if job['dataset'] in ['mar31-gymnastics', 'apr16-gymnastics']:
        job['gymnastics_dataset_location'] = os.path.join(
            directory, 'mar-31-2019')
    elif job['dataset'] in [
            'jun1full-gymnastics', 'jun1sub6-gymnastics',
            'jun1sub25-gymnastics', 'jun1sub50-gymnastics',
            'jun1sub75-gymnastics'
    ]:
        job['gymnastics_dataset_location'] = os.path.join(
            directory, 'jun-01-2019')
    elif job['dataset'] in ['vlog']:
        job['vlog_train'] = os.path.join(directory, 'VLOG', 'vlog_train.txt')
        job['vlog_test'] = os.path.join(directory, 'VLOG', 'vlog_test.txt')
        job['vlog_val'] = os.path.join(directory, 'VLOG', 'vlog_val.txt')        
    else:
        raise

    time = 48
    if 'time' in job:
        time = job.pop('time')
        
    if local_comet_dir:
        job['local_comet_dir'] = local_comet_dir    

    num_gpus = job.pop('num_gpus')
    job['gpu-id'] = ','.join([str(k) for k in range(num_gpus)])
    job['checkpoint'] = os.path.join(directory, 'supercons', 'tc-ckpts', job['name'])

    num_cpus = job.pop('num_cpus')
    job['workers'] = min(int(2.5 * num_gpus), num_cpus - num_gpus)
    job['workers'] = max(job['workers'], 12)

    gb = job.pop('gb')
    memory_per_node = min(gb, 500)

    print(job)

    flagstring = " --counter %d" % counter
    for key, value in job.items():
        if type(value) == bool:
            if value == True:
                flagstring += " --%s" % key
        elif _is_int(value):
            flagstring += ' --%s %d' % (key, value)
        elif _is_float(value):
            flagstring += ' --%s %.6f' % (key, value)
        else:
            flagstring += ' --%s %s' % (key, value)

    jobname = "tc.%s" % job['name']
    jobcommand = "python train_video_cycle_simple.py %s" % flagstring
    print(jobcommand)

    slurmfile = os.path.join(slurm_scripts, jobname + '.slurm')

    with open(slurmfile, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name=%s\n" % jobname)
        # f.write("#SBATCH --qos=batch\n")
        f.write("#SBATCH --mail-type=END,FAIL\n")
        f.write("#SBATCH --mail-user=%s\n" % email)
        f.write("#SBATCH --gres=ntasks-per-node=1\n")
        f.write("#SBATCH --cpus-per-task=%d\n" % num_cpus)
        f.write("#SBATCH --time=%d:00:00\n" % time)
        f.write("#SBATCH --gres=gpu:%d\n" % num_gpus)
        f.write("#SBATCH --mem=%dG\n" % memory_per_node)
        # f.write("#SBATCH --constraint=gpu_12gb\n")
        f.write("#SBATCH --nodes=1\n")

        f.write("#SBATCH --output=%s\n" %
                os.path.join(slurm_logs, jobname + ".out"))
        f.write("#SBATCH --error=%s\n" %
                os.path.join(slurm_logs, jobname + ".err"))

        f.write("module purge" + "\n")
        module_load(f)
        f.write("source activate supercons\n")
        f.write("SRCDIR=%s\n" % code_directory)
        f.write("cd ${SRCDIR}\n")
        f.write(jobcommand + "\n")

    s = "sbatch %s" % os.path.join(slurm_scripts, jobname + ".slurm")
    os.system(s)


def cims_run_batch(job, counter, email, code_directory):

    def module_load(f):
        f.write("module load cuda-10.0\n")

    directory = '/misc/kcgscratch1/ChoGroup/resnick/spaceofmotion'
    slurm_logs = os.path.join(directory, "slurm_logs")
    slurm_scripts = os.path.join(directory, "slurm_scripts")
    _run_batch(job,
               counter,
               slurm_logs,
               slurm_scripts,
               module_load,
               directory,
               email,
               code_directory)


def hpc_run_batch(job, counter, email, code_directory):

    def module_load(f):
        f.write("module load anaconda3/5.3.1\n")
        f.write("module load cuda/9.0.176\n")
        f.write("module load cudnn/9.0v7.3.0.29\n")

    directory = '/beegfs/cr2668/spaceofmotion'
    slurm_logs = os.path.join(directory, "slurm_logs")
    slurm_scripts = os.path.join(directory, "slurm_scripts")
    _run_batch(job,
               counter,
               slurm_logs,
               slurm_scripts,
               module_load,
               directory,
               email,
               code_directory)


def fb_run_batch(job, counter, email, code_directory):

    def module_load(f):
        f.write("module load cuda/10.0\n")
        # f.write("module load anaconda3/5.3.1\n")
        # f.write("module load cuda/9.0.176\n")
        # f.write("module load cudnn/9.0v7.3.0.29\n")

    directory = '/checkpoint/cinjon/spaceofmotion'
    slurm_logs = os.path.join(directory, 'supercons', 'tc-slurm_logs')
    slurm_scripts = os.path.join(directory, 'supercons', 'tc-slurm_scripts')
    comet_dir = os.path.join(directory, 'supercons', 'tc-comet')
    _run_batch(job,
               counter,
               slurm_logs,
               slurm_scripts,
               module_load,
               directory,
               email,
               code_directory,
               local_comet_dir=comet_dir)
