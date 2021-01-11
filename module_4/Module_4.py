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
