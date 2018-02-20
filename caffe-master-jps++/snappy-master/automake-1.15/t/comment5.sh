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

# Test for PR/280.
# (Automake should complain about trailing backslashes in comments.)

. test-init.sh

cat >> configure.ac <<'EOF'
AC_OUTPUT
EOF

cat > Makefile.am << 'EOF'
all-local:
	@echo ${var}

# a comment with backslash \


var = foo
EOF

$ACLOCAL
AUTOMAKE_fails
grep '^Makefile.am:5: error: blank line following trailing backslash' stderr


## Here is a second test because head comments are
## handled differently in Automake 1.5.

cat > Makefile.am << 'EOF'
# a comment with backslash \


all-local:
	@echo ${var}

var = foo
EOF

AUTOMAKE_fails
grep '^Makefile.am:2: error: blank line following trailing backslash' stderr


## Make sure we print an 'included' stack on errors.

echo 'include Makefile.inc'> Makefile.am
cat > Makefile.inc << 'EOF'
# a comment with backslash \

EOF

AUTOMAKE_fails
grep '^Makefile.inc:2: error: blank line following trailing backslash' stderr
grep '^Makefile.am:1: .*included from here' stderr
grep -v '^Makefile.am:1: .*error:' stderr


## Make sure backslashes are still allowed within a comment.
## This usually happens when commenting out a Makefile rule.

cat > Makefile.am << 'EOF'
all-local:
	@echo ${var}

# a comment with backslash \
# but terminated by a line without backslash

var = foo
EOF

$AUTOMAKE
