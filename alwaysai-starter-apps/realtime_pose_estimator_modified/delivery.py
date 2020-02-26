import requests
def send(message):
    """Construct a post request and deliver"""
    try:
        url = "https://api.twilio.com/2010-04-01/Accounts/AC52acc29bbf0d2d1c3d28fe30232d8394/Messages.json"
        user =
        token =
        data = {
            "To": ,
            "From": ,
            "Body": message#"Domestic Situation identified. Notifying Authorities."
        }
        r = requests.post(url, data=data, auth=(user, token))
        # r = requests.post(url=url, data=data)
        #r = requests.post(url=url, data=data, files=files)
        if r.status_code != 200:
            # logger.log('delivery.py: send: post response code: {}, reason: {}'.format(
            #         r.status_code, r.reason), should_log)
            return r
        # logger.log('delivery.py: send: successful', should_log)
        return r
    except requests.exceptions.ConnectionError:
        # logger.log('Connection refused for connection to: {}'.format(url), should_log)
        return 'Connection refused'
    except:
        # logger.log('Unknown post request error for connection to: {}'.format(url), should_log)
        return 'Unknown post request error'
