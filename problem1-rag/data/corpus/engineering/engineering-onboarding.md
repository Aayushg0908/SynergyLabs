---
title: Engineering Onboarding Guide
department: engineering
doc_type: engineering
updated: 2026-05-30
---

# Engineering Onboarding Guide

Welcome to Kestrel Labs engineering. This guide covers your first two weeks.

## Day one

- Collect your laptop from IT on the 2nd floor of Building A. Engineers receive a 16-inch
  laptop with 64 GB of RAM; if you work on the simulation team you may request a desktop
  workstation with a GPU instead.
- Enroll a hardware security key. Two keys are issued: one to carry, one to keep at home.
  Password-only login is disabled for all engineering systems.
- Join the #eng-announcements and #deploys Slack channels.

## Your onboarding buddy

Every new engineer is paired with an onboarding buddy from a different team for their
first **90 days**. Buddies meet with you at least once a week. Your manager will
introduce you on day one.

## Development environment

The monorepo is hosted on the internal Git server. Clone it with the `kdev bootstrap`
command, which installs the toolchain, pre-commit hooks and the local simulator. A clean
bootstrap takes about 40 minutes on the standard laptop.

All code changes need at least **two approving reviews**, one of which must come from a
code owner of the files changed. Changes to flight-control code additionally require
approval from the flight-safety team.

## First tasks

By the end of week one, you should have merged a small change (a "starter ticket" tagged
`good-first-issue`). By the end of week two, you should have shadowed one on-call shift.

## On-call

Engineers join the on-call rotation after **8 weeks**. Shifts are one week long, Monday
10:00 to Monday 10:00 local time. On-call engineers receive a stipend of $500 per week of
primary on-call and $200 per week of secondary on-call. If you are paged between
midnight and 06:00, you may start work late the next day without using leave.

## Learning budget

Each engineer has an annual learning budget of **$2,000** for courses, books and
conferences. Conference travel itself is covered by the travel and expense policy, not
the learning budget.
