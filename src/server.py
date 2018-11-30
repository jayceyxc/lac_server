#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018-11-29 21:06
# @Author  : yuxuecheng
# @Contact : yuxuecheng@xinluomed.com
# @Site    : 
# @File    : server.py
# @Software: PyCharm
# @Description lac分词web服务器

import numpy as np
import tornado.ioloop
import tornado.web
from tornado import httpserver
import argparse
import reader
import paddle
from paddle import fluid
import infer
import logging
import os


def init_logging(filename):
    filename = os.path.join("logs", filename)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(filename)s:%(lineno)-4d: %(message)s",
                        datefmt="%m-%d %H:%M:%S",
                        filename=filename,
                        filemode="a")


def parse_args():
    parser = argparse.ArgumentParser("Run inference.")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=5,
        help='The size of a batch. (default: %(default)d)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='./conf/model',
        help='A path to the model. (default: %(default)s)'
    )
    parser.add_argument(
        '--test_data_dir',
        type=str,
        default='./data/test_data',
        help='A directory with test data files. (default: %(default)s)'
    )
    parser.add_argument(
        "--word_dict_path",
        type=str,
        default="./conf/word.dic",
        help="The path of the word dictionary. (default: %(default)s)"
    )
    parser.add_argument(
        "--label_dict_path",
        type=str,
        default="./conf/tag.dic",
        help="The path of the label dictionary. (default: %(default)s)"
    )
    parser.add_argument(
        "--word_rep_dict_path",
        type=str,
        default="./conf/q2b.dic",
        help="The path of the word replacement Dictionary. (default: %(default)s)"
    )
    args = parser.parse_args()
    return args


def cut_sentence(args, line):
    id2word_dict = reader.load_dict(args.word_dict_path)
    word2id_dict = reader.load_reverse_dict(args.word_dict_path)

    id2label_dict = reader.load_dict(args.label_dict_path)
    label2id_dict = reader.load_reverse_dict(args.label_dict_path)
    q2b_dict = reader.load_dict(args.word_rep_dict_path)
    test_data = paddle.batch(
        reader.parse_line_wrapper(content=line,
                                  word2id_dict=word2id_dict,
                                  word_replace_dict=q2b_dict),
        batch_size=args.batch_size)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(args.model_path, exe)
        for data in test_data():
            full_out_str = ""
            word_idx = infer.to_lodtensor([x[0] for x in data], place)
            word_list = [x[1] for x in data]
            (crf_decode,) = exe.run(inference_program,
                                    feed={"word": word_idx},
                                    fetch_list=fetch_targets,
                                    return_numpy=False)
            lod_info = (crf_decode.lod())[0]
            np_data = np.array(crf_decode)
            assert len(data) == len(lod_info) - 1
            for sen_index in xrange(len(data)):
                assert len(data[sen_index][0]) == lod_info[
                    sen_index + 1] - lod_info[sen_index]
                word_index = 0
                outstr = ""
                cur_full_word = ""
                cur_full_tag = ""
                words = word_list[sen_index]
                for tag_index in xrange(lod_info[sen_index],
                                        lod_info[sen_index + 1]):
                    cur_word = words[word_index]
                    cur_tag = id2label_dict[str(np_data[tag_index][0])]
                    if cur_tag.endswith("-B") or cur_tag.endswith("O"):
                        if len(cur_full_word) != 0:
                            outstr += cur_full_word.encode('utf8') + "/" + cur_full_tag.encode('utf8') + " "
                        cur_full_word = cur_word
                        cur_full_tag = infer.get_real_tag(cur_tag)
                    else:
                        cur_full_word += cur_word
                    word_index += 1
                outstr += cur_full_word.encode('utf8') + "/" + cur_full_tag.encode('utf8') + " "
                outstr = outstr.strip()
                full_out_str += outstr + "\n"
            logging.info('分词结果：{0}'.format(full_out_str.strip()))
            return full_out_str.strip()


class LacHandler(tornado.web.RequestHandler):
    def post(self, *args, **kwargs):
        logging.info(main_args)
        if self.request.arguments.has_key('data'):
            logging.info(self.get_argument('data').encode(encoding='utf8'))
            result = cut_sentence(main_args, self.get_argument('data').encode(encoding='utf8'))
            self.write(result)
        else:
            logging.info('没有数据')
            self.write('没有数据')


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/lac", LacHandler),
    ])


if __name__ == "__main__":
    main_args = parse_args()
    init_logging('lac_server.log')
    logging.info(main_args)
    app = make_app()
    http_server = httpserver.HTTPServer(app)
    http_server.listen(8888)
    tornado.ioloop.IOLoop.current().start()
