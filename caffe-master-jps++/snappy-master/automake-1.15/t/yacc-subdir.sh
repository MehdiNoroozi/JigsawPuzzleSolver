#! /bin/sh
# Copyright (C) 2002-2014 Free Software Foundation, Inc.
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

# Test for subdir parsers.

required='cc yacc'

. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_CC
AC_PROG_YACC
AC_OUTPUT
END

cat > Makefile.am << 'END'
AUTOMAKE_OPTIONS = subdir-objects
bin_PROGRAMS = foo/foo
foo_foo_SOURCES = foo/parse.y
AM_YFLAGS = -d

.PHONY: obj
obj: foo/parse.$(OBJEXT)

.PHONY: test1 test2
test1: foo/parse.$(OBJEXT)
	test -f foo/parse.c
	test -f foo/parse.$(OBJEXT)
test2: foo/parse2.$(OBJEXT)
	test -f foo/parse2.c
	test -f foo/parse2.$(OBJEXT)
END

mkdir foo

cat > foo/parse.y << 'END'
%{
int yylex () { return 0; }
void yyerror (char *s) {}
%}
%%
foobar : 'f' 'o' 'o' 'b' 'a' 'r' {};
END

$ACLOCAL
$AUTOCONF
$AUTOMAKE -a

mkdir sub
cd sub

../configure
$MAKE test1

# Aside of the rest of this test, let's see if we can recover from
# parse.h removal.
test -f foo/parse.h
rm -f foo/parse.h
$MAKE foo/parse.h
test -f foo/parse.h

# Make sure foo/parse.h is not updated, unless when needed.
$sleep
touch ../foo/parse.y
$MAKE obj
is_newest ../foo/parse.y foo/parse.h
$sleep
sed 's/%%/%token TOKEN\n%%/g' ../foo/parse.y >../foo/parse.yt
mv -f ../foo/parse.yt ../foo/parse.y
$MAKE obj
is_newest foo/parse.h ../foo/parse.y

# Now, adds another parser to test ylwrap.

cd ..

# Sleep some to make sure timestamp of Makefile.am will change.
$sleep

cp foo/parse.y foo/parse2.y
cat >> Makefile.am << 'END'
EXTRA_foo_foo_SOURCES = foo/parse2.y
END

$AUTOMAKE -a
test -f ./ylwrap

cd sub
# Regenerate Makefile (automatic in GNU Make, but not in other Makes).
./config.status
$MAKE test2

:
