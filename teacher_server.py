"""
Standalone teacher server: loads InternVL3-38B on GPU0 and serves
feedback requests from RL training ranks over a TCP socket.

Launch BEFORE torchrun:
    CUDA_VISIBLE_DEVICES=0 python teacher_server.py \
        --model_path ./pretrained/InternVL3-38B \
        --port 55123
"""
import argparse
import json
import socket
import struct
import os
import sys
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from navsim.agents.recogdrive.utils.internvl_preprocess import load_image


def recv_msg(conn):
    raw = b""
    while len(raw) < 4:
        chunk = conn.recv(4 - len(raw))
        if not chunk:
            return None
        raw += chunk
    length = struct.unpack(">I", raw)[0]
    data = b""
    while len(data) < length:
        chunk = conn.recv(length - len(data))
        if not chunk:
            return None
        data += chunk
    return json.loads(data.decode())


def send_msg(conn, obj):
    data = json.dumps(obj).encode()
    conn.sendall(struct.pack(">I", len(data)) + data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./pretrained/InternVL3-38B")
    parser.add_argument("--port", type=int, default=55123)
    args = parser.parse_args()

    print(f"[TeacherServer] Loading {args.model_path} on cuda:0 ...")
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map={"": "cuda:0"},
    ).eval()
    if hasattr(model, "vision_model"):
        model.vision_model.to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False
    )
    gen_cfg = dict(max_new_tokens=256, do_sample=False)
    print(f"[TeacherServer] Ready. Listening on port {args.port}")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(16)

    while True:
        conn, addr = srv.accept()
        try:
            req = recv_msg(conn)
            if req is None:
                continue
            image_path  = req["image_path"]
            prompt_text = req["prompt_text"]
            pv = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
            with torch.no_grad():
                feedback = model.chat(
                    tokenizer, pv, "<image>\n" + prompt_text, gen_cfg
                )
            send_msg(conn, {"feedback": feedback})
        except Exception as e:
            send_msg(conn, {"feedback": "", "error": str(e)})
        finally:
            conn.close()


if __name__ == "__main__":
    main()
