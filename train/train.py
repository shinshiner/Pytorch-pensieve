from __future__ import division
import torch
import torch.nn.functional as F
from utils import ensure_shared_grads
from model import agentNET
from torch.autograd import Variable
from env import *
import random

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
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

def train(rank, args, shared_model, optimizer, all_cooked_time, all_cooked_bw):
    torch.manual_seed(args.seed + rank)
    env = Environment(all_cooked_time=all_cooked_time,
                    all_cooked_bw=all_cooked_bw,
                    random_seed=rank
                      )

    model = agentNET()
    model.train()

    time_stamp = 0
    uploadtime = 0
    end_of_video = True
    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY
    entropy_weights = [5, 1, 1, 0.5, 0.5, 0.1, 0.1, 0.1, 0.01] + [0.01] * 1000
    entropy_weight = 0.01   # default entropy weight

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
            bit_rate  = random.randint(0,5)
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

        values = []
        log_probs = []
        rewards = []
        entropies = []

        entropy_weight = 20000 / (uploadtime + 4000)
        # entropy_weight = 0.05
        # entropy_weight = entropy_weights[int(uploadtime // 1000)]

        while True:
            if args.gpu:
                value, logit, (hx, cx) = model((Variable(state.unsqueeze(0)).cuda(), (hx, cx)))
            else:
                value, logit, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx)))

            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            if args.gpu:
                action = prob.multinomial().data.cpu()
                action.view(-1, 1)
                log_prob = log_prob.gather(1, Variable(action.cuda()))
            else:
                action = prob.multinomial().data
                action.view(-1, 1)
                log_prob = log_prob.gather(1, Variable(action))

            bit_rate = action.numpy()[0][0]

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
            state[0][4][S_LEN - 1] = (np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K)[action.numpy()[0][0]]  # mega byte
            state[0][5][S_LEN - 1] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
            # state = torch.from_numpy(np.array([state, ])).float()
            last_bit_rate = bit_rate

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY
                break

        # update the network
        R = torch.zeros(1, 1)
        if not end_of_video:
            if args.gpu:
                value, _, _ = model((Variable(state.unsqueeze(0).cuda()), (hx, cx)))
            else:
                value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data

        if args.gpu:
            values.append(Variable(R.cuda()))
            R = Variable(R.cuda())
        else:
            values.append(Variable(R))
            R = Variable(R)

        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)

        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            if args.gpu:
                delta_t = rewards[i] + args.gamma * \
                                       values[i + 1].data.cpu() - values[i].data.cpu()
            else:
                delta_t = rewards[i] + args.gamma * \
                                       values[i + 1].data - values[i].data

            gae = gae * args.gamma * args.tau + delta_t

            if args.gpu:
                policy_loss = policy_loss - \
                              log_probs[i] * Variable(gae.cuda()) - entropy_weight * entropies[i]
            else:
                policy_loss = policy_loss - \
                              log_probs[i] * Variable(gae) - entropy_weight * entropies[i]

        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()

        if args.gpu:
            model = model.cpu()

        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        ensure_shared_grads(model, shared_model)
        optimizer.step()

        uploadtime += 1
        if uploadtime % 1000 == 0 and rank == 1:
            print('---> after {0} steps <---'.format(uploadtime * args.workers))