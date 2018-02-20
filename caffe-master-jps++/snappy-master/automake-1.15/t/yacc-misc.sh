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

# Some simple tests of ylwrap functionality.

required='cc yacc'
. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_CC
AC_PROG_YACC
AC_OUTPUT
END

cat > Makefile.am << 'END'
bin_PROGRAMS = foo bar
foo_SOURCES = parse.y foo.c
bar_SOURCES = bar.y foo.c
END

# First parser.
cat > parse.y << 'END'
%{
int yylex () { return 0; }
void yyerror (char *s) {}
%}
%%
foobar : 'f' 'o' 'o' 'b' 'a' 'r' {};
END

# Second parser.
cat > bar.y << 'END'
%{
int yylex () { return 0; }
void yyerror (char *s) {}
%}
%%
fubar : 'f' 'o' 'o' 'b' 'a' 'r' {};
END

cat > foo.c << 'END'
int main (void)
{
  return 0;
}
END

$ACLOCAL
$AUTOCONF
$AUTOMAKE -a

test -f ylwrap

mkdir sub
cd sub

../configure
$MAKE

grep '^#.*/sub/\.\./' bar.c && exit 1
grep '^#.*/sub/\.\./' parse.c && exit 1

# Make distclean must not erase bar.c nor parse.c (by GNU standards) ...
$MAKE distclean
test -f bar.c
test -f parse.c
# ... but maintainer-clean should.
../configure
$MAKE maintainer-clean
test ! -e bar.c
test ! -e parse.c

:
