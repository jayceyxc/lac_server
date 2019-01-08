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
import ctypes
import re

so = ctypes.cdll.LoadLibrary
lac_lib = so('./lib/liblac.so')
print("cut_sentence")
lac_lib.freeme.argtypes = ctypes.c_void_p,
lac_lib.freeme.restype = None
lac_lib.cut.restype = ctypes.c_void_p
lac_lib.lexer.restype = ctypes.c_void_p
lac_lib.posseg.restype = ctypes.c_void_p


def extract_chinese(s):
    line = s.strip().decode('utf-8', 'ignore')  # 处理前进行相关的处理，包括转换成Unicode等
    p2 = re.compile(ur'[^\u4e00-\u9fa5]')  # 中文的编码范围是：\u4e00到\u9fa5
    zh = " ".join(p2.split(line)).strip()
    zh = re.sub(' +', ' ', zh)
    out_str = zh  # 经过相关处理后得到中文的文本
    return out_str


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
    parser.add_argument(
        "--conf_path",
        type=str,
        default="./conf/",
        help="The path of the configure file Dictionary. (default: %(default)s)"
    )
    args = parser.parse_args()
    return args


def cut_sentence_python(args, line):
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


def cut_sentence_cpp(line, conf_dir='../conf'):
    line = extract_chinese(line)
    max_result_num = 10240
    result = lac_lib.cut(conf_dir, max_result_num, line.encode('utf8'))
    if result is None:
        temp_result_num = max_result_num * 10
        result = lac_lib.cut(conf_dir, temp_result_num, line.encode('utf8'))
        if result is None:
            temp_result_num = temp_result_num * 10
            result = lac_lib.cut(conf_dir, temp_result_num, line.encode('utf8'))
            if result is None:
                temp_result_num = temp_result_num * 10
                result = lac_lib.cut(conf_dir, temp_result_num, line.encode('utf8'))
            else:
                logging.error('the content too long')
                logging.warning(type(result))
    cut_result = ctypes.cast(result, ctypes.c_char_p).value
    if cut_result is not None:
        logging.info("return result: " + cut_result)
    else:
        logging.info("return None")
        cut_result = ''
    lac_lib.freeme(result)
    return cut_result


def lexer_sentence_cpp(line, conf_dir='../conf'):
    max_result_num = 10240
    result = lac_lib.lexer(conf_dir, max_result_num, line)
    if result is None:
        temp_result_num = max_result_num * 10
        result = lac_lib.lexer(conf_dir, temp_result_num, line)
        if result is None:
            temp_result_num = temp_result_num * 10
            result = lac_lib.lexer(conf_dir, temp_result_num, line)
            if result is None:
                temp_result_num = temp_result_num * 10
                result = lac_lib.lexer(conf_dir, temp_result_num, line)
            else:
                logging.error('the content too long')
                logging.warning(type(result))
    lexer_result = ctypes.cast(result, ctypes.c_char_p).value
    logging.info("return result: " + lexer_result)
    lac_lib.freeme(result)
    # logging.info(lac_lib.sum(1, 2))

    return lexer_result


def posseg_sentence_cpp(line, conf_dir='../conf'):
    max_result_num = 10240
    result = lac_lib.posseg(conf_dir, max_result_num, line)
    if result is None:
        temp_result_num = max_result_num * 10
        result = lac_lib.posseg(conf_dir, temp_result_num, line)
        if result is None:
            temp_result_num = temp_result_num * 10
            result = lac_lib.posseg(conf_dir, temp_result_num, line)
            if result is None:
                temp_result_num = temp_result_num * 10
                result = lac_lib.posseg(conf_dir, temp_result_num, line)
            else:
                logging.error('the content too long')
                logging.warning(type(result))
    posseg_result = ctypes.cast(result, ctypes.c_char_p).value
    logging.info("return result: " + posseg_result)
    lac_lib.freeme(result)
    # logging.info(lac_lib.sum(1, 2))

    return posseg_result


class CutHandler(tornado.web.RequestHandler):
    def post(self, *args, **kwargs):
        logging.info(main_args)
        if self.request.arguments.has_key('data'):
            logging.info(self.get_argument('data').encode(encoding='utf8'))
            result = cut_sentence_cpp(self.get_argument('data').encode(encoding='utf8'), conf_dir=main_args.conf_path)
            self.write(result)
        else:
            logging.info('没有数据')
            self.write('没有数据')


class LexerHandler(tornado.web.RequestHandler):
    def post(self, *args, **kwargs):
        logging.info(main_args)
        if self.request.arguments.has_key('data'):
            logging.info(self.get_argument('data').encode(encoding='utf8'))
            result = lexer_sentence_cpp(self.get_argument('data').encode(encoding='utf8'), conf_dir=main_args.conf_path)
            self.write(result)
        else:
            logging.info('没有数据')
            self.write('没有数据')


class PossegHandler(tornado.web.RequestHandler):
    def post(self, *args, **kwargs):
        logging.info(main_args)
        if self.request.arguments.has_key('data'):
            logging.info(self.get_argument('data').encode(encoding='utf8'))
            result = posseg_sentence_cpp(self.get_argument('data').encode(encoding='utf8'), conf_dir=main_args.conf_path)
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
        (r"/cut", CutHandler),
        (r"/lexer", LexerHandler),
        (r"/posseg", PossegHandler),
    ])


if __name__ == "__main__":
    main_args = parse_args()
    init_logging('lac_server.log')
    logging.info(main_args)
    app = make_app()
    http_server = httpserver.HTTPServer(app)
    http_server.listen(8888)
    tornado.ioloop.IOLoop.current().start()
