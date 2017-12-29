import os
import sys
import signal
import subprocess
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from pyvirtualdisplay import Display
from time import sleep

def timeout_handler(signum, frame):
	raise Exception("Timeout")

abr_algo = sys.argv[1]
run_time = int(sys.argv[2])
exp_id = sys.argv[3]
url = 'http://localhost/' + 'myindex_' + abr_algo + '.html'

# timeout signal
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(run_time + 30)

with open('a', 'w') as ff:
	ff.write('start\n')
	
try:
	# copy over the chrome user dir
	default_chrome_user_dir = '../abr_browser_dir/chrome_data_dir'
	chrome_user_dir = '/tmp/chrome_user_dir_real_exp_' + abr_algo
	os.system('rm -r ' + chrome_user_dir)
	os.system('cp -r ' + default_chrome_user_dir + ' ' + chrome_user_dir)

	with open('a', 'a') as ff:
		ff.write('starting abr servers\n')
	
	# start abr algorithm server
	command = 'exec /usr/bin/python ../rl_server/rl_server.py ' + exp_id
	
	proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
	sleep(2)

	# to not display the page in browser
	display = Display(visible=1, size=(1280,700))
	display.start()
	
	# initialize chrome driver
	options=Options()
	chrome_driver = '../abr_browser_dir/chromedriver'
	options.add_argument('--user-data-dir=' + chrome_user_dir)
	options.add_argument('--ignore-certificate-errors')
	driver=webdriver.Chrome(chrome_driver, chrome_options=options)
	driver.set_window_size(1280, 700)
	
	# run chrome
	with open('a', 'a') as ff:
		ff.write('getting url\n')
		ff.flush()
	driver.set_page_load_timeout(10)
	driver.get(url)
	with open('a', 'a') as ff:
		ff.write('get url\n')
		ff.flush()
	
	sleep(run_time)
	
	driver.quit()
	#display.stop()
	
	# kill abr algorithm server
	proc.send_signal(signal.SIGINT)
	# proc.kill()
	
	print('done')
	
except Exception as e:
	try:
		print('stop')
		display.stop()
	except:
		pass
	try:
		print('quit')
		driver.quit()
	except:
		pass
	try:
		print('send_signal')
		proc.send_signal(signal.SIGINT)
	except:
		pass
	
	print(e)

