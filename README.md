# COVIDetector
An anonymous crowdsourced mobile solutions to track and contain COVID-19 possible transmission. 

It’s a self-reporting and self-checking App letting the users: 
1. know if they have been inadvertently in prolonged close proximity with any other diagnosed user (eg. in < 2 meters for > 15 minutes)
2. or if a user is diagnosed then alert other users that they are in elevated risk of infection, without compromising the reporting users' identity

How could we do it?
- The proximity are estimated by Bluetooth signal strength (RSSN). 
- The user’s privacy and anonymity are protected by differential privacy mechanisms (eg. bloom filter based)

An unique feature of this app is that even if you have never installed this app before, once installed it can still show you immediately if you have elevated risks.

Other privacy features are:
- *NO data will ever been uploaded to the cloud platform or shared with other users if a user is just checking his/her risk.*
- Only when a COVID positive user sends an alert via the cloud platform, the differential privacy hashed bluetooth ids filter that contain the list of at-risk users will be send to the cloud, but *the reporting user's id will NOT be saved at cloud platform nor will it ever be shared with other users*. 
