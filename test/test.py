from __future__ import division
import torch
import torch.nn.functional as F
import load_trace
from model import agentNET
from torch.autograd import Variable
import fixed_env as Env
import numpy as np
import random

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
NUM_AGENTS = 1
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 0  # default video quality without agent
LOG_FILE = './results/log_sim_better_rl'
NN_MODEL = ''

def test():
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace()

    torch.manual_seed(233)
    env = Env.Environment(all_cooked_time=all_cooked_time,
                      all_cooked_bw=all_cooked_bw,
                      random_seed=50
                      )
    log_path = LOG_FILE + '_' + all_file_names[env.trace_idx]
    log_file = open(log_path, 'w')

    model = agentNET()
    model.eval()
    saved_state = torch.load(NN_MODEL)
    model.load_state_dict(saved_state)

    time_stamp = 0
    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY
    video_count = 0
    finish = False

    while True:
        cx = Variable(torch.zeros(1, 96))
        hx = Variable(torch.zeros(1, 96))

        state = np.zeros([S_INFO, S_LEN])
        # do an default action
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            env.get_video_chunk(DEFAULT_QUALITY)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        state = np.zeros([S_INFO, S_LEN])
        for i in range(S_LEN):
            # do an default action
            bit_rate = random.randint(0, 5)
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # get new state
            state[0][i] = VIDEO_BIT_RATE[last_bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1][i] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2][i] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3][i] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4][i] = (np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K)[DEFAULT_QUALITY]  # mega byte
            state[5][i] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

            last_bit_rate = bit_rate
        state = torch.from_numpy(np.array([state, ])).float()

        reward_sum = 0

        while True:
            value, logit, (hx, cx) = model((Variable(state.unsqueeze(0)),(hx, cx)))
            prob = F.softmax(logit)
            action = prob.max(1)[1].data
            # action_value = random.random()
            # for i in range(A_DIM):
            #     action_value = action_value - prob.data[0][i]
            #     if action_value <= 0:
            #         action = torch.IntTensor([i])
            #         break

            bit_rate = action.numpy()[0]

            # do an action
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # -- linear reward --
            # reward is video quality - rebuffer penalty - smoothness
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                               VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

            # get new state
            for i in range(S_INFO):
                for j in range(S_LEN - 1):
                    state[0][i][j] = state[0][i][j + 1]
            # state = np.zeros(S_INFO)
            state[0][0][S_LEN - 1] = VIDEO_BIT_RATE[last_bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[0][1][S_LEN - 1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[0][2][S_LEN - 1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[0][3][S_LEN - 1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[0][4][S_LEN - 1] = (np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K)[action.numpy()[0]]  # mega byte
            state[0][5][S_LEN - 1] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
            # state = torch.from_numpy(np.array([state, ])).float()
            last_bit_rate = bit_rate
            reward_sum += reward

            log_file.write(str(time_stamp / M_IN_K) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            if end_of_video:
                log_file.write('\n')
                log_file.close()

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY

                print("video count", video_count)
                video_count += 1

                if video_count >= len(all_file_names):
                    finish = True
                    break

                log_path = LOG_FILE + '_' + all_file_names[env.trace_idx]
                log_file = open(log_path, 'w')

        if finish:
            break

test()
