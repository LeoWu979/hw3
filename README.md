# hw3

### setup<br> 
- uLCD : tx <-> D1, rx <-> D0, reset <-> D2.<br>
- use "sudo screen /dev/ttyACM*" to open screen on terminal.
- LED1 indicates Gesture_UI mode.
- LED2 indicates Tilt_Detection mode.
- LED3 indicates Initialize gravity vector or not.

### Gesture_UI mode : <br>
- Use "/Gesture_UI/run 1" to start the Gesture_UI mode.<br>
- You will see LED1 to be turned on.
- This UI has four Threshold angle : 25, 30, 35, 40.<br>
- Use "Ring" gesture to select up and use "down" gesture to select down.<br>
- Current mode will both show on the uLCD and terminal.<br>
- Press the "USER_BUTTON" to confirm the Threshold Angle, and you will see your selection on screen.<br>
- Then system go back to RPC loop and LED1 turn off.<br>

### Tilt_Detection mode : <br>
- Use "/Tilt_Detection/run 1" to start the Tilt_Detection mode.<br>
- You will see LED2 and LED3 both to be turned on.<br>
- Please put your mbed board on the table, then press the "USER_BUTTON" to complete initialization of gravity vector, then LED3 turn off.<br>
- You can start to tilt the mbed board arbitrarily, the current tilt angle will show on uLCD.<br>
- If current angle is greater than theshold angle, PC will show the event on screen(my detect period is 500ms).<br>
- After 10 times of events, PC will send the command to stop this mode and back to RPC loop.<br>
