#! /bin/sh
# Copyright (C) 2003-2014 Free Software Foundation, Inc.
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

# Regression test for install-recursive appearing in a non recursive Makefile.
# Report from Bruno Haible.

. test-init.sh

cat > Makefile.am << 'END'
noinst_SCRIPTS = hostname
include_HEADERS = gettext-po.h
BUILT_SOURCES = po-hash-gen.c
END

$ACLOCAL
$AUTOMAKE
grep 'install-recursive' Makefile.in && exit 1

:
