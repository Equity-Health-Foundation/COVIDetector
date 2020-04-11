# COVIDetector
An anonymous crowdsourced mobile solutions to track and contain COVID-19 possible transmission. 

It’s a self-reporting and self-checking App letting the users: 
1. know if they have been inadvertently in prolonged close proximity with any other diagnosed user (eg. in < 2 meters for > 15 minutes)
2. or if a user is diagnosed then alert other users that they are in elevated risk of infection, without compromising the reporting users' identity

How could we do it?
- The proximity are estimated by Bluetooth signal strength (RSSN). 
- The user’s privacy and anonymity are protected by differential privacy mechanisms (bloom filter based)

An unique feature of this app is that even if you have never installed this app before, once installed it can still show you immediately if you have elevated risks.
