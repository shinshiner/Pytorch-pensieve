from __future__ import division
import torch
import torch.nn.functional as F
from utils import setup_logger
from model import agentNET
from torch.autograd import Variable
from env import *
import numpy as np
import time
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
TEST_INTERVAL = 70

def test(args, shared_model, all_cooked_time, all_cooked_bw):
    logger = setup_logger("test_log",  "./logs/test_log")

    torch.manual_seed(args.seed)
    env = Environment(all_cooked_time=all_cooked_time,
                      all_cooked_bw=all_cooked_bw,
                      random_seed=50
                      )

    model = agentNET()
    model.eval()

    test_time = 0
    reward_num = 0
    max_reward = 0

    time_stamp = 0
    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    while True:
        model.load_state_dict(shared_model.state_dict())
        if args.gpu:
            model = model.cuda()
            cx = Variable(torch.zeros(1, 96).cuda())
            hx = Variable(torch.zeros(1, 96).cuda())
        else:
            cx = Variable(torch.zeros(1, 96))
            hx = Variable(torch.zeros(1, 96))

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

        while True:
            if args.gpu:
                value, logit, (hx, cx) = model((Variable(state.unsqueeze(0)).cuda(), (hx, cx)))
            else:
                value, logit, (hx, cx) = model((Variable(state.unsqueeze(0)),(hx, cx)))

            prob = F.softmax(logit)
            if args.gpu:
                action = prob.max(1)[1].data.cpu()
            else:
                action = prob.max(1)[1].data

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
            reward_num += reward

            if end_of_video:
                # print(env.trace_idx)
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY
                test_time += 1
                break

        if test_time % TEST_INTERVAL == 0:
            logger.info('reward = ' + str(reward_num / TEST_INTERVAL))

            if reward_num >= max_reward:
                max_reward = reward_num
                # save better model
                logger.info('save_better')
                if args.gpu:
                    model = model.cpu()
                state_to_save = model.state_dict()
                torch.save(state_to_save, "./results/Nbisieve.dat")
                if args.gpu:
                    model = model.cuda()
            # save model after a long training
            if test_time % (TEST_INTERVAL * 10) == 0:
                logger.info('save_long')
                if args.gpu:
                    model = model.cpu()
                state_to_save = model.state_dict()
                torch.save(state_to_save, "./results/Nbisieve_long.dat")
                if args.gpu:
                    model = model.cuda()
            # save good model
            if reward_num > 45 * TEST_INTERVAL:
                logger.info('save_more_than_45')
                if args.gpu:
                    model = model.cpu()
                state_to_save = model.state_dict()
                torch.save(state_to_save, "./results/Nbisieve%d.dat" % (reward_num / TEST_INTERVAL))
                if args.gpu:
                    model = model.cuda()

            reward_num = 0
            time.sleep(30)
