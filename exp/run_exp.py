import subprocess
import numpy as np

RUN_SCRIPT = 'run_video.py'
RANDOM_SEED = 42
RUN_TIME = 280  # sec
REPEAT_TIME = 3

def main():
	np.random.seed(RANDOM_SEED)
	with open('./chrome_retry_log', 'w') as log:
		log.write('chrome retry log\n')
		log.flush()

		for rt in range(REPEAT_TIME):
			while True:
				script = 'python ' + RUN_SCRIPT + ' ' + \
						  'RL' + ' ' + str(RUN_TIME) + ' ' + str(rt)

				proc = subprocess.Popen(script,
						  stdout=subprocess.PIPE,
						  stderr=subprocess.PIPE,
						  shell=True)

				(out, err) = proc.communicate()

				if out == 'done\n':
					break
				else:
					log.write('RL' + '_' + str(rt) + '\n')
					log.write(out + '\n')
					log.flush()

if __name__ == '__main__':
	main()
