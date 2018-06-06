# -*- coding: utf-8 -*-
import socket
from Settings.arguments import TexasHoldemArgument as Argument


class ACPCNetworkCommunication:
    def __init__(self):
        self.server = None
        self.port = None
        self.connection = None
        self.socket_file = None

    def connect(self, server, port):
        self.server = server or Argument.server
        self.port = port or Argument.port

        self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connection.connect(address=(server, port))

        self.socket_file = self.connection.makefile()
        self._hand_shake()

    def _hand_shake(self):
        self.send_line("VERSION:2.0.0")

    def send_line(self, line):
        message = (line + "\r\n").encode()
        self.connection.send(message)

    def get_line(self):
        message = self.socket_file.readline()
        line = message.strip("\n").strip("\r")
        return line

    def close(self):
        self.connection.close()
