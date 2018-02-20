#! /bin/sh
# Copyright (C) 2001-2014 Free Software Foundation, Inc.
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

# Test for bug in conditionals.

. test-init.sh

cat >> configure.ac << 'END'
dnl Define a macro with the same name as the conditional to exhibit
dnl any underquoted bug.
AC_DEFUN([COND1], ["some'meaningless;characters`])
AM_CONDITIONAL([COND1], [false])
AC_CONFIG_FILES([foo/Makefile])
AC_CONFIG_FILES([bar/Makefile])
AC_OUTPUT
END

cat > Makefile.am << 'END'
AUTOMAKE_OPTIONS = no-dependencies
CC = false

SUBDIRS = foo
if COND1
SUBDIRS += bar
endif

# Small example from the manual.
bin_PROGRAMS = hello
hello_SOURCES = hello-common.c
if COND1
hello_SOURCES += hello-cond1.c
else
hello_SOURCES += hello-generic.c
endif

.PHONY: test
test: distdir
	test -f $(distdir)/foo/Makefile.am
	test -f $(distdir)/bar/Makefile.am
	test -f $(distdir)/hello-common.c
	test -f $(distdir)/hello-cond1.c
	test -f $(distdir)/hello-generic.c
END

mkdir foo bar

: > foo/Makefile.am
: > bar/Makefile.am
: > hello-common.c
: > hello-cond1.c
: > hello-generic.c

$ACLOCAL
$AUTOCONF
grep "meaningless;characters" configure && exit 1
$AUTOMAKE
./configure
$MAKE test

:
