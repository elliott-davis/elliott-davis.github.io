---
title: "Storming, Forming and Norming"
date: 2022-09-15T20:40:24-05:00
author: "Elliott Davis"
authorTwitter: "libsysguy"
cover: "images/herewego.gif"
showFullContent: false
---

Invariably, every new team goes through the storming, forming and norming phases. There is the initial honeymoon period. Everyone is excited, nobody wants to step on toes and we all crave quick success. Then, some one gets annoyed, the project slows down and shit just hits the fan. How the hell, are you as an engineering lead supposed to handle the shit storm?

I have hit the storming phase several times in my career (the curse of job hopping). I have started developing some tenants of bootstrapping a new team that will hopefully make this storming phase more manageable:

# The tenants of bootstrapping a new engineering team

* We map product value to users, through user stories. Technical details can be added as tasks or as definitions of done
    * This prevents the refactor kings and queens on your team from going down the rabbit hole
* We communicate openly in slack and zoom. If you need help, ask for a pair or ask a team for help in slack
    * :shockedpikachu: People are often uncomfortable not looking knowledgeable. Create a culture of leaning and acceptance of failure.
* We raise issues early. If something is bugging you or you see a gap, raise the issue as soon as possible
    * Don't let people get to a point of boiling. Set a space for people to raise issues and surface problems early
* Rules are enforced as code. Whether it’s through CI linting or editorconfig, we need not nit-pick each other
    * When you see something that bugs you, AUTOMATE FIXING IT. Code review can suck. Make the CI system or your editor the enemy, not your teammates.

I'm still learning, I'm sure there are more tenants that will help teams get through the storming period so feel free to drop in a PR and update this list