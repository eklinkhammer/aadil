#!/usr/bin/env perl

while(<STDIN>) {

    # Parses vector
    my @var = split(':', $_);
    my @vec = split(' ', $var[1]);
    shift @vec;
    pop @vec;
    print join(',', @vec), "\n"
};
