#! /bin/sh
# Copyright (C) 2004-2014 Free Software Foundation, Inc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Make sure 'compile' preserves spaces in its arguments.

am_create_testdir=empty
. test-init.sh

get_shell_script compile

# -o 'a  c' should not be stripped because 'a  c' is not an object
# (it does not matter whether touch creates ./-- or not)
./compile touch a.o -- -o 'a  c' a.c
test -f 'a  c'
test -f ./-o
test -f a.o
test -f a.c

rm -f 'a  c' ./-o a.o a.c

./compile touch a.o -- -o 'a  c.o' a.c
test -f 'a  c.o'
test ! -e ./-o
test ! -e a.o
test -f a.c

# Make sure 'compile' works for .obj too.
./compile touch a.obj -- -o ac.obj a.c
test ! -e a.obj
test ac.obj

:
