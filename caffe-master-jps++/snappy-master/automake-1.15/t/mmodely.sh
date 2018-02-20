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

# Verify that intermediate files are only built from Yacc and Lex
# sources in maintainer mode.
# From Derek R. Price.

required=cc
. test-init.sh

cat >> configure.ac << 'END'
AM_MAINTAINER_MODE
AC_PROG_CC
AM_PROG_LEX
AC_PROG_YACC
AC_OUTPUT
END

cat > Makefile.am <<'END'
YACC = false
LEX = false
bin_PROGRAMS = zardoz
zardoz_SOURCES = zardoz.y joe.l
LDADD = @LEXLIB@
END

# The point of this test is that it is not dependent on a working lex
# or yacc.
cat > joe.c <<EOF
int joe (int arg)
{
    return arg * 2;
}
EOF
# On systems which link in libraries non-lazily and whose linkers
# complain about unresolved symbols by default, such as Solaris, an
# yylex function needs to be defined to avoid an error due to an
# unresolved symbol.
cat > zardoz.c <<EOF
int joe (int arg);
int yylex (void)
{
    return 0;
}
int main (int argc, char **argv)
{
    return joe (argc);
}
EOF

# Ensure a later timestamp for our Lex & Yacc sources.
$sleep
: > joe.l
: > zardoz.y

$ACLOCAL
$AUTOCONF
$AUTOMAKE -a

./configure
$MAKE

cat >myyacc.sh <<'END'
#! /bin/sh
echo "$@" >y.tab.c
END
cat >mylex.sh <<'END'
echo "$@" >lex.yy.c
END
chmod +x myyacc.sh mylex.sh
PATH=$(pwd)$PATH_SEPARATOR$PATH; export PATH

# "make maintainer-clean; ./configure; make" should always work,
# per GNU Standard.
$MAKE maintainer-clean
./configure
run_make YACC=myyacc.sh LEX=mylex.sh LEX_OUTPUT_ROOT=lex.yy zardoz.c joe.c
$FGREP zardoz.y zardoz.c
$FGREP joe.l joe.c

:
