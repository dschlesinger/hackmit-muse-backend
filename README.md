Inspiration
What inspired you to create this project?
The feeling when you're trying to study and you have a deadline, but you can’t lock in. I don’t know exactly what causes this crippling condition, but one cure is well-placed and tasteful music. Additionally, convenience is the bread and butter of locking in. We decided to use an EEG for constant background monitoring so the user can keep their mind on more important things. We also used AR glasses so the user can quickly check what songs are playing and receive alerts without the need for a distracting iPhone screen.

What it does
Explain what your project does.
It's hard to study, especially without the right environment. Music is a great way to set the environment, but randomly sampling Spotify recommendations can kill the mood.  That's why we use an EEG to detect whether the user is concentrated, relaxed, or stressed and customize a suno playlist to best fit it. Additionally, we can send the currently playing song and alerts to the heads-up display in our Mentra AR glasses to remind the user to take a break if fatigue reaches certain levels.

How we built it
Explain how you built your project.
We built a multi-stage signal processing system using Python and the Muse EEG headset:
Gathering Data: Used Lab Streaming Layer (LSL) to gather 256Hz EEG data from frontal channels
Real-Time filtering: Implemented notch filters to remove 60Hz power line noise
Feature Extraction: Computed band powers for Delta, Theta, Alpha, and Beta frequencies using FFT analysis
We repurposed EEG artifacts as intentional input:
Single and Double Blink Recognition: implemented temporal pattern matching to detect two blinks within 2 seconds for song skipping
We created smart queueing integrated with Suno’s music API
State-Driven Generation: Brain states trigger specific music prompts (ambient for relaxation, electronic for focus, calming for stress)
Background Generation: New songs generate while current music plays to maintain seamless transitions
AR Glasses Integration: Sends song metadata and notifications to display glasses

Technical stack:
EEG Processing: NumPy, SciPy for signal processing, PyLSL for data streaming
Music Generation: Suno API integration with VLC-based audio playback
AR Integration: HTTP API endpoints for glasses communications

Individual Contributions
Explain how the work was divided among teammates.
We worked from the outside in. Our project consists of a pipeline that goes from the EEG sensor through to the Suno API, and then outputs all legwork to display glasses. After working through how we were to lay out that pipeline, we decided to start at each end – one of us on glasses, one of us on EEG – and work towards the Suno API to connect the two halves of our idea.

Breakdown of work:
Jaxon – EEG, brain state monitoring, and song queue management
Denali – Setup Glasses, created routing between Python, glasses, and Suno api

Challenges we ran into
Explain any challenges you ran into.
During:
EEG Connection → The python library provided by Muse to connect with the headband hasn’t been updated for the headband we have, so we had to redo some of the backend dependencies for newer Bluetooth protocols
Blink/Brain State Monitoring → For the blinking to skip feature, had to fine-tune the parameters of blink and brain state detection to ensure consistent brain state results and responsive playback control → Finding the sensitivity sweet spot ensures there aren’t any false positive blinks
Pregeneration and Queue → It takes a significant amount of time for Suno to get back with the generated songs, and we had some trouble ensuring our pregenerated buffer was long enough and robust enough to last until the generated songs came back
Routing EEG -> Glasses: Issues with maintaining a connection, would sometimes time out unexpectedly. Added redundancy to deal with timeouts
Suno API -> Figuring out what to do with the delay between a generation request and fulfillment, ended up pre-generating 2 starting songs


Accomplishments that we're proud of
Explain any accomplishments you're proud of.
We are both very new to hardware and came into the hackathon wanting to try new things. We do computational biology most of the time, lol. I am proud that we persisted and had a good time. The feeling of the whole pipeline finally working was amazing!


What we learned
Explain what you learned.
Jaxon – How to work with real-time EEG data streams, Understanding brain wave frequency bands (Delta, Theta, Alpha, Beta) and their psychological correlations, Managing multiple concurrent processes (EEG processing, blink detection, music generation, playback, etc.), Eye blink detection using frontal EEG channels (AF7, AF8) to turn artifacts into features


Denali – I learned how to route our data from our Python EEG monitor to our glasses (frontend?). Websockets used to scare me, and now they do even more. And how to interpret/use eeg data.


What's next for our project
Given more time, in what ways could you further expand on your project?
We would like a front interface, a website, to manage the song queues. Use the glasses bitmap function for a more detailed UI. Improve the EEG monitoring and look into new metrics.
