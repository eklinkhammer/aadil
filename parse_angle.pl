#!/usr/bin/env perl

while(<STDIN>) {

    # Parses Angle
    my @var = split(' ', $_);
    my @state1 = split(',', $var[2]);
    my @state2 = split(',', $var[5]);
    print  $state1[2]. ' ', $state2[2], "\n"
};
