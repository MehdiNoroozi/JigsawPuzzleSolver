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

# Check whether double colon rules work.  The Unix V7 make manual
# mentions double-colon rules, but POSIX does not.  They seem to be
# supported by all Make implementation as far as we can tell. This test
# case is a spy: we want to detect if there exist implementations where
# these do not work.  We might use these rules to simplify the rebuild
# rules (instead of the $? hack).

# Tom Tromey write:
# | In the distant past we used :: rules extensively.
# | Fran?ois convinced me to get rid of them:
# |
# | Thu Nov 23 18:02:38 1995  Tom Tromey  <tromey@cambric>
# | [ ... ]
# |         * subdirs.am: Removed "::" rules
# |         * header.am, libraries.am, mans.am, texinfos.am, footer.am:
# |         Removed "::" rules
# |         * scripts.am, programs.am, libprograms.am: Removed "::" rules
# |
# |
# | I no longer remember the rationale for this.  It may have only been a
# | belief that they were unportable.

# On a related topic, the Autoconf manual has the following text:
# |     'VPATH' and double-colon rules
# |           Any assignment to 'VPATH' causes Sun 'make' to only execute
# |           the first set of double-colon rules.  (This comment has been
# |           here since 1994 and the context has been lost.  It's probably
# |           about SunOS 4.  If you can reproduce this, please send us a
# |           test case for illustration.)

# We already know that overlapping ::-rule like
#
#   a :: b
#   	echo rule1 >> $@
#   a :: c
#   	echo rule2 >> $@
#   a :: b c
#   	echo rule3 >> $@
#
# do not work equally on all platforms.  It seems that in all cases
# Make attempts to run all matching rules.  However at least GNU Make,
# NetBSD Make, and FreeBSD Make will detect that $@ was updated by the
# first matching rule and skip remaining matches (with the above
# example that means that unless 'a' was declared PHONY, only "rule1"
# will be appended to 'a' if both b and c have changed).  Other
# implementations like OSF1 Make and HP-UX Make do not perform such a
# check and execute all matching rules whatever they do ("rule1",
# "rule2", abd "rule3" will all be appended to 'a' if b and c have
# changed).

# So it seems only non-overlapping ::-rule may be portable.  This is
# what we check now.

. test-init.sh

cat >Makefile <<\EOF
a :: b
	echo rule1 >> $@
a :: c
	echo rule2 >> $@
EOF

touch b c
$sleep
: > a
$MAKE
test x"$(cat a)" = x
$sleep
touch b
$MAKE
test "$(cat a)" = "rule1"
# Ensure a is strictly newer than b, so HP-UX make does not execute rule2.
$sleep
: > a
$sleep
touch c
$MAKE
test "$(cat a)" = "rule2"

# Unfortunately, the following is not portable to FreeBSD/NetBSD/OpenBSD
# make, see explanation above.

#: > a
#$sleep
#touch b c
#$MAKE
#grep rule1 a
#grep rule2 a

:
