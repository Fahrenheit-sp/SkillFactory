#!/usr/bin/env python
# coding: utf-8

-- 4.1 (Moscow, Uljanovsk)
SELECT a.city,
       count(a.airport_code) airports_count
FROM dst_project.airports a
GROUP BY city
HAVING count(a.airport_code) > 1

-- 4.2.1 (6)
SELECT count(DISTINCT f.status) status_count
FROM dst_project.flights f

-- 4.2.2 (58)
SELECT count(DISTINCT f.flight_id) flights_count
FROM dst_project.flights f
WHERE f.status = 'Departed'

-- 4.2.3. (402)
SELECT s.aircraft_code,
       count(s.seat_no)
FROM dst_project.seats s
GROUP BY s.aircraft_code
HAVING s.aircraft_code = '773'

-- 4.2.4 (74227)
SELECT count(f.flight_id)
FROM dst_project.flights f
WHERE ((f.scheduled_arrival BETWEEN '2017.04.01' AND '2017.09.01')
       OR (f.scheduled_departure BETWEEN '2017.04.01' AND '2017.09.01'))
  AND (f.status = 'Arrived')
  
-- 4.3.1 (437)
SELECT count(f.flight_id)
FROM dst_project.flights f
WHERE f.status = 'Cancelled'

-- 4.3.2.
SELECT count(a.aircraft_code)
FROM dst_project.aircrafts a
WHERE a.model like '%Boeing%' -- replace with required model name

-- 4.3.3 (Europe, Asia)
SELECT 'Europe' continent,
                count(a.airport_code)
FROM dst_project.airports a
WHERE a.timezone like '%Europe%'
UNION ALL
SELECT 'Asia' continent,
              count(a.airport_code)
FROM dst_project.airports a
WHERE a.timezone like '%Asia%'
UNION ALL
SELECT 'Australia' continent,
                   count(a.airport_code)
FROM dst_project.airports a
WHERE a.timezone like '%Australia%'
ORDER BY 2

-- 4.3.4 (157751)
SELECT f.flight_id,
       (f.actual_arrival - f.scheduled_arrival) span
FROM dst_project.flights f
WHERE f.actual_arrival IS NOT NULL
  AND f.scheduled_arrival IS NOT NULL
GROUP BY f.flight_id
ORDER BY 2 DESC
LIMIT 1

-- 4.4.1 (14.08.2016)
SELECT min(f.scheduled_departure)
FROM dst_project.flights f
WHERE f.scheduled_arrival IS NOT NULL

-- 4.4.2 (530)
SELECT max(f.scheduled_arrival - f.scheduled_departure)
FROM dst_project.flights f
WHERE f.scheduled_arrival IS NOT NULL
  AND f.scheduled_departure IS NOT NULL
  
-- 4.4.3 (DME-UUS)
  SELECT f.departure_airport,
       f.arrival_airport,
       max(f.scheduled_arrival - f.scheduled_departure) max_length
FROM dst_project.flights f
WHERE f.scheduled_arrival IS NOT NULL
  AND f.scheduled_departure IS NOT NULL
GROUP BY f.departure_airport,
         f.arrival_airport
ORDER BY max_length DESC

-- 4.4.4 (128)
SELECT avg(f.scheduled_arrival - f.scheduled_departure)
FROM dst_project.flights f
WHERE f.scheduled_arrival IS NOT NULL
  AND f.scheduled_departure IS NOT NULL
  
-- 4.5.1 (Economy 85)
SELECT s.fare_conditions,
       count(s.fare_conditions) seats_count
FROM dst_project.seats s
WHERE s.aircraft_code = 'SU9'
GROUP BY s.fare_conditions
ORDER BY seats_count DESC

-- 4.5.2 (3400)
SELECT min(b.total_amount)
FROM dst_project.bookings b

-- 4.5.3 (2A)
SELECT p.seat_no
FROM dst_project.tickets t
LEFT JOIN dst_project.boarding_passes p ON t.ticket_no = p.ticket_no
WHERE t.passenger_id = '4313 788533'

-- 5.1.1 (426)
SELECT count(f.flight_id)
FROM dst_project.airports a
LEFT JOIN dst_project.flights f ON a.airport_code = f.arrival_airport
WHERE a.city = 'Anapa'
  AND (f.actual_arrival BETWEEN '2017.01.01' AND '2017.12.31')
  
-- 5.1.2 (127)
SELECT COUNT (f.flight_id)
FROM dst_project.flights f
LEFT JOIN dst_project.airports a ON a.airport_code = f.departure_airport
WHERE a.city = 'Anapa'
  AND f.actual_departure BETWEEN '2017-01-01 00:00:00' AND '2017-02-28 23:59:59'
  OR f.actual_departure BETWEEN '2017-12-01 00:00:00' AND '2017-12-31 23:59:59'
       
-- 5.1.3 (1)
SELECT count(f.flight_id)
FROM dst_project.airports a
LEFT JOIN dst_project.flights f ON a.airport_code = f.departure_airport
WHERE a.city = 'Anapa'
  AND f.status = 'Cancelled'
  
-- 5.1.4 (453)
SELECT count(*)
FROM dst_project.airports a
LEFT JOIN dst_project.flights f ON a.airport_code = f.departure_airport
WHERE a.city = 'Anapa' -- 843 fligts from Anapa total

SELECT count(*)
FROM dst_project.airports a
LEFT JOIN dst_project.flights f ON a.airport_code = f.arrival_airport
WHERE a.city = 'Moscow'
  AND f.departure_airport = 'AAQ' -- 396 flights from Anapa to Moscow. Difference is 843 - 396 = 453

-- 5.1.5 (Boeing 737-300)
SELECT p.model,
       count(DISTINCT s.seat_no) seats_count
FROM dst_project.airports a
LEFT JOIN dst_project.flights f ON a.airport_code = f.departure_airport
LEFT JOIN dst_project.seats s ON s.aircraft_code = f.aircraft_code
LEFT JOIN dst_project.aircrafts p ON p.aircraft_code = s.aircraft_code
WHERE a.city = 'Anapa'
GROUP BY p.model
ORDER BY seats_count DESC
LIMIT 1


-- CSV EXPORT

SELECT f.flight_id,
       f.arrival_airport,
       plane.model,
       (f.actual_arrival - f.actual_departure) duration,
       count(pass.seat_no) places_taken,
       sum(b.total_amount) amount
FROM dst_project.flights f
JOIN dst_project.aircrafts plane ON f.aircraft_code = plane.aircraft_code
JOIN dst_project.boarding_passes pass ON f.flight_id = pass.flight_id
JOIN dst_project.tickets t ON pass.ticket_no = t.ticket_no
JOIN dst_project.bookings b ON t.book_ref = b.book_ref
WHERE f.departure_airport = 'AAQ'
  AND (date_trunc('month', f.scheduled_departure) in ('2017-01-01',
                                                      '2017-02-01',
                                                      '2017-12-01'))
  AND f.status not in ('Cancelled')
GROUP BY f.flight_id,
         f.arrival_airport,
         plane.model
